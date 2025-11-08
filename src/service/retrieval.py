"""Utilities for retrieving code snippets from an RDF knowledge graph using the trained weighting model."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import os

import hashlib
import os

import torch
from rdflib import Graph, Literal, Namespace, RDF, URIRef, XSD
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from peft import PeftModel
except ImportError as exc:  # pragma: no cover - inference requires peft
    raise ImportError("peft must be installed to load the DoRA adapter") from exc

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain

TW = Namespace("http://example.org/train/vocab/")


@dataclass
class CodeCandidate:
    uri: str
    file_uri: str
    file_path: str
    name: str
    kind: str
    language: str
    version: str
    code: str
    description: str = ""
    cluster: int | None = None

    def as_passage(self) -> str:
        header = (
            f"Block {self.name} ({self.kind}) located in {self.file_path} "
            f"for {self.language} / {self.version}."
        )
        parts = [header]
        if self.description:
            parts.append(f"Description:\n{self.description}")
        parts.append(f"Code:\n{self.code}")
        return "\n\n".join(parts)


@dataclass
class CodeGraph:
    candidates: List[CodeCandidate]
    adjacency: Dict[str, List[str]]
    by_uri: Dict[str, CodeCandidate]
    rdf_graph: Graph
    clusters: Dict[int, List[str]] | None = None  # cluster_id -> list of uris
    cluster_centroids: Dict[int, np.ndarray] | None = None
    border_nodes: Dict[int, List[str]] | None = None  # cluster_id -> list of border uris
    embeddings: np.ndarray | None = None


@dataclass
class EnrichmentPreview:
    cluster_id: int
    suggestions: List[Tuple[CodeCandidate, float]]


def save_graph(graph: Graph, path: Path) -> None:
    """Serialize the RDF graph back to Turtle format at the given path."""
    graph.serialize(destination=str(path), format="turtle")


def slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    import re
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "node"


def build_embeddings(candidates: List[CodeCandidate], tokenizer: AutoTokenizer, model: AutoModelForSequenceClassification, device: str) -> np.ndarray:
    """Build embeddings for candidates using the model's base for semantic vectors."""
    embeddings = []
    model.to(device)
    for cand in candidates:
        inputs = tokenizer(cand.as_passage(), return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            # Use base model for embeddings (CLS token)
            outputs = model.base_model(**inputs, output_hidden_states=True)
            emb = outputs.hidden_states[-1][:, 0, :].cpu().numpy().flatten()
        embeddings.append(emb)
    return np.array(embeddings)


def detect_communities(code_graph: CodeGraph) -> Dict[int, List[str]]:
    """Detect communities in the graph using Louvain method."""
    G = nx.Graph()
    for uri, neighbors in code_graph.adjacency.items():
        for neigh in neighbors:
            G.add_edge(uri, neigh)
    communities = nx.community.louvain_communities(G)
    return {i: list(comm) for i, comm in enumerate(communities)}


def hybrid_clustering(code_graph: CodeGraph, embeddings: np.ndarray, n_clusters: int = 5) -> Tuple[Dict[int, List[str]], Dict[int, np.ndarray], Dict[int, List[str]]]:
    """Combine graph communities with TF-IDF clustering for hybrid sub-regions."""
    graph_communities = detect_communities(code_graph)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    tfidf_clusters = {}
    for idx, label in enumerate(labels):
        uri = code_graph.candidates[idx].uri
        tfidf_clusters.setdefault(label, []).append(uri)
    centroids = kmeans.cluster_centers_

    # Hybrid: Refine graph communities with TF-IDF sub-clusters
    hybrid_clusters = {}
    cid = 0
    for comm_uris in graph_communities.values():
        indices = [code_graph.candidates.index(code_graph.by_uri[uri]) for uri in comm_uris]
        sub_matrix = embeddings[indices]
        if len(indices) > 1:
            sub_kmeans = KMeans(n_clusters=min(3, len(indices)), random_state=42)
            sub_labels = sub_kmeans.fit_predict(sub_matrix)
            for sub_idx, label in enumerate(sub_labels):
                global_idx = indices[sub_idx]
                hybrid_clusters.setdefault(cid + label, []).append(code_graph.candidates[global_idx].uri)
        else:
            hybrid_clusters[cid] = comm_uris
        cid += 10

    hybrid_centroids = {}
    for hid, uris in hybrid_clusters.items():
        indices = [code_graph.candidates.index(code_graph.by_uri[uri]) for uri in uris]
        if indices:
            hybrid_centroids[hid] = np.mean(embeddings[indices], axis=0)

    border_nodes = compute_border_nodes_hybrid(hybrid_clusters, hybrid_centroids, embeddings, code_graph)
    return hybrid_clusters, hybrid_centroids, border_nodes


def compute_border_nodes_hybrid(clusters: Dict[int, List[str]], centroids: Dict[int, np.ndarray], embeddings: np.ndarray, code_graph: CodeGraph, top_border: int = 3) -> Dict[int, List[str]]:
    """Compute border nodes for hybrid clusters."""
    border_nodes = {}
    for cluster_id, uris in clusters.items():
        indices = [code_graph.candidates.index(code_graph.by_uri[uri]) for uri in uris]
        distances = []
        for other_id, other_centroid in centroids.items():
            if other_id == cluster_id:
                continue
            for idx in indices:
                dist = cosine_similarity([embeddings[idx]], [other_centroid])[0][0]
                distances.append((idx, dist))
        distances.sort(key=lambda x: x[1])
        border_nodes[cluster_id] = [code_graph.candidates[idx].uri for idx, _ in distances[:top_border]]
    return border_nodes


def assign_cluster(centroids: Dict[int, np.ndarray], new_vector: np.ndarray) -> int:
    """Assign a new vector to the closest cluster centroid."""
    if not centroids:
        return -1
    similarities = {cid: cosine_similarity([new_vector], [cent])[0][0] for cid, cent in centroids.items()}
    return max(similarities, key=similarities.get)


def select_candidates_for_enrichment(
    code_graph: CodeGraph,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: str,
    new_candidate: CodeCandidate,
    top_k: int = 10
) -> List[CodeCandidate]:
    """Select top_k candidates for enrichment by comparing to cluster + borders."""
    new_embedding = build_embeddings([new_candidate], tokenizer, model, device)[0]
    cluster_id = assign_cluster(code_graph.cluster_centroids or {}, new_embedding)

    clusters = code_graph.clusters or {}
    border_map = code_graph.border_nodes or {}

    if cluster_id == -1:
        candidates_in_cluster = list(code_graph.candidates)
        border_candidates: List[CodeCandidate] = []
    else:
        cluster_members = clusters.get(cluster_id, [])
        candidates_in_cluster = [code_graph.by_uri[uri] for uri in cluster_members if uri in code_graph.by_uri]
        border_candidates = []
        for cid, uris in border_map.items():
            if cid != cluster_id:
                border_candidates.extend([code_graph.by_uri[uri] for uri in uris if uri in code_graph.by_uri])

    all_candidates = candidates_in_cluster + border_candidates

    similarities = []
    for cand in all_candidates:
        idx = code_graph.candidates.index(cand)
        cand_embedding = code_graph.embeddings[idx]
        sim = cosine_similarity([new_embedding], [cand_embedding])[0][0]
        similarities.append((cand, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return [cand for cand, _ in similarities[:top_k]]


def preview_enrichment(
    code_graph: CodeGraph,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: str,
    new_candidate: CodeCandidate,
    *,
    max_length: int = 256,
    doc_penalty: float = 0.0,
    top_k: int = 10,
    threshold: float | None = None,
) -> List[Tuple[CodeCandidate, float]]:
    """Estimate link probabilities for a new candidate without mutating the graph."""
    selected = select_candidates_for_enrichment(
        code_graph,
        tokenizer,
        model,
        device,
        new_candidate,
        top_k=top_k,
    )
    scored: List[Tuple[CodeCandidate, float]] = []
    for cand in selected:
        score = compute_score(
            tokenizer,
            model,
            cand.as_passage(),
            new_candidate,
            max_length=max_length,
            device=device,
            doc_penalty=doc_penalty,
        )
        prob = 1 / (1 + np.exp(-score))
        if threshold is None or prob >= threshold:
            scored.append((cand, prob))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def enrich_graph(
    code_graph: CodeGraph,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: str,
    new_candidate: CodeCandidate,
    query: str = "",
    max_length: int = 256,
    doc_penalty: float = 0.0,
    threshold: float = 0.5,
    top_k: int = 10
) -> None:
    """Enrich the graph by adding connections from new_candidate to selected candidates."""
    from rdflib import URIRef, Literal, XSD
    TW = Namespace("http://example.org/train/vocab/")
    
    # First, add the new node itself to the graph
    new_uri = URIRef(new_candidate.uri)
    file_uri = URIRef(new_candidate.file_uri)
    
    # Add node metadata
    code_graph.rdf_graph.add((new_uri, RDF.type, TW.CodeBlock))
    code_graph.rdf_graph.add((new_uri, TW.name, Literal(new_candidate.name)))
    code_graph.rdf_graph.add((new_uri, TW.code, Literal(new_candidate.code)))
    code_graph.rdf_graph.add((new_uri, TW.kind, Literal(new_candidate.kind)))
    code_graph.rdf_graph.add((new_uri, TW.belongsToFile, file_uri))
    
    if new_candidate.description:
        code_graph.rdf_graph.add((new_uri, TW.description, Literal(new_candidate.description)))
    
    # Add file metadata if not exists
    if not list(code_graph.rdf_graph.triples((file_uri, RDF.type, TW.File))):
        code_graph.rdf_graph.add((file_uri, RDF.type, TW.File))
        code_graph.rdf_graph.add((file_uri, TW.path, Literal(new_candidate.file_path)))
    
    # Add language and version metadata
    language_uri = URIRef(f"http://example.org/train/language/{new_candidate.language.lower()}")
    version_uri = URIRef(f"http://example.org/train/version/{slugify(new_candidate.version)}")
    
    code_graph.rdf_graph.add((new_uri, TW.belongsToLanguage, language_uri))
    code_graph.rdf_graph.add((new_uri, TW.belongsToVersion, version_uri))
    
    if not list(code_graph.rdf_graph.triples((language_uri, RDF.type, TW.Language))):
        code_graph.rdf_graph.add((language_uri, RDF.type, TW.Language))
        code_graph.rdf_graph.add((language_uri, TW.label, Literal(new_candidate.language)))
    
    if not list(code_graph.rdf_graph.triples((version_uri, RDF.type, TW.Version))):
        code_graph.rdf_graph.add((version_uri, RDF.type, TW.Version))
        code_graph.rdf_graph.add((version_uri, TW.label, Literal(new_candidate.version)))
    
    # Now compute and add connections
    probabilities = preview_enrichment(
        code_graph,
        tokenizer,
        model,
        device,
        new_candidate,
        max_length=max_length,
        doc_penalty=doc_penalty,
        top_k=top_k,
        threshold=threshold,
    )
    connected = [cand for cand, _ in probabilities]

    # Add connection triples
    for cand in connected:
        cand_uri = URIRef(cand.uri)
        code_graph.rdf_graph.add((new_uri, TW.relatedTo, cand_uri))
        weight = next(prob for c, prob in probabilities if c == cand)
        code_graph.rdf_graph.add((new_uri, TW.weight, Literal(weight, datatype=XSD.double)))

    # Save the graph
    graph_path = Path("src/train_weighting/dataset/new_graph.ttl")
    save_graph(code_graph.rdf_graph, graph_path)


def is_documentation(candidate: CodeCandidate) -> bool:
    """Check if the candidate is a documentation block."""
    return candidate.kind.lower() in ["documentation", "docstring", "comment"] or "doc" in candidate.name.lower()


def compute_score(
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    query: str,
    candidate: CodeCandidate,
    *,
    max_length: int,
    device: str,
    doc_penalty: float,
) -> float:
    inputs = tokenizer(
        query,
        candidate.as_passage(),
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        raw_score = model(**inputs).logits.squeeze().item()
    penalty = doc_penalty if doc_penalty > 0 and is_documentation(candidate) else 0.0
    return raw_score - penalty


def load_candidates(graph_path: Path, adapter_path: Path, base_model: str, device: str) -> CodeGraph:
    tokenizer, model = load_model(adapter_path, base_model)
    model.to(device)
    graph = Graph().parse(graph_path)
    candidates: List[CodeCandidate] = []
    for block_uri in graph.subjects(RDF.type, TW.CodeBlock):
        name_literal = graph.value(block_uri, TW.name)
        code_literal = graph.value(block_uri, TW.code)
        kind_literal = graph.value(block_uri, TW.kind)
        file_uri = graph.value(block_uri, TW.belongsToFile)
        version_uri = graph.value(block_uri, TW.belongsToVersion)
        language_uri = graph.value(block_uri, TW.belongsToLanguage)
        description_literal = graph.value(block_uri, TW.description)
        if not all((name_literal, code_literal, file_uri, version_uri, language_uri)):
            continue
        file_path_literal = graph.value(file_uri, TW.path)
        language_label = graph.value(language_uri, TW.label)
        version_label = graph.value(version_uri, TW.label)
        candidate = CodeCandidate(
            uri=str(block_uri),
            file_uri=str(file_uri),
            file_path=str(file_path_literal) if file_path_literal else "",
            name=str(name_literal),
            kind=str(kind_literal) if kind_literal else "section",
            language=str(language_label) if language_label else "",
            version=str(version_label) if version_label else "",
            code=str(code_literal),
            description=str(description_literal) if description_literal else "",
        )
        candidates.append(candidate)

    adjacency: Dict[str, List[str]] = {
        candidate.uri: [other.uri for other in candidates if other.uri != candidate.uri]
        for candidate in candidates
    }

    by_uri = {candidate.uri: candidate for candidate in candidates}

    embeddings = build_embeddings(candidates, tokenizer, model, device) if candidates else np.empty((0,))

    code_graph = CodeGraph(
        candidates=candidates,
        adjacency=adjacency,
        by_uri=by_uri,
        rdf_graph=graph,
        clusters={},
        cluster_centroids={},
        border_nodes={},
        embeddings=embeddings,
    )

    if len(candidates) >= 2:
        try:
            clusters, cluster_centroids, border_nodes = hybrid_clustering(
                code_graph,
                embeddings,
                n_clusters=min(5, len(candidates)),
            )
            code_graph.clusters = clusters
            code_graph.cluster_centroids = cluster_centroids
            code_graph.border_nodes = border_nodes
        except ValueError:
            # KMeans can raise when all samples are identical; keep defaults.
            pass

    return code_graph


def load_model(adapter_path: Path, base_model: str) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return tokenizer, model


def score_candidates(
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    query: str,
    candidates: Iterable[CodeCandidate],
    max_length: int = 256,
    device: str = "cpu",
    doc_penalty: float = 0.0,
) -> Tuple[List[Tuple[CodeCandidate, float]], Dict[str, float]]:
    results: List[Tuple[CodeCandidate, float]] = []
    score_map: Dict[str, float] = {}
    for candidate in candidates:
        score = compute_score(
            tokenizer,
            model,
            query,
            candidate,
            max_length=max_length,
            device=device,
            doc_penalty=doc_penalty,
        )
        results.append((candidate, score))
        score_map[candidate.uri] = score
    results.sort(key=lambda item: item[1], reverse=True)
    return results, score_map


def resolve_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank code snippets from an RDF graph using the trained weighting model",
    )
    parser.add_argument(
        "query",
        help="Natural-language description of the desired code",
    )
    parser.add_argument(
        "--graph",
        default="src/train_weighting/dataset/new_graph.ttl",
        help="Path to the RDF/Turtle knowledge graph",
    )
    parser.add_argument(
        "--adapter",
        default="artifacts/weighting-dora",
        help="Directory containing the trained DoRA adapter",
    )
    parser.add_argument(
        "--base-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Base cross-encoder model identifier",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top matching snippets to display",
    )
    parser.add_argument(
        "--device",
        help="Torch device override (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum token length for encoder inputs",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of graph expansion steps to follow from the top results",
    )
    parser.add_argument(
        "--top-p",
        type=int,
        default=3,
        help="Beam width used for the tree search expansion",
    )
    parser.add_argument(
        "--doc-penalty",
        type=float,
        default=1.5,
        help="Penalty applied to documentation-only blocks to reduce their rank",
    )
    parser.add_argument(
        "--enable-logging",
        action="store_true",
        help="Enable query logging for co-occurrence analysis",
    )
    parser.add_argument(
        "--log-path",
        default="logs/query_log.jsonl",
        help="Path to query log file",
    )
    return parser.parse_args()


def beam_search_paths(
    graph: CodeGraph,
    roots: List[CodeCandidate],
    score_map: Dict[str, float],
    steps: int,
    beam_width: int,
    *,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    query: str,
    max_length: int,
    device: str,
    doc_penalty: float,
) -> Tuple[List[CodeCandidate], List[List[Tuple[CodeCandidate, float]]]]:
    if not roots:
        return [], []

    def refresh_score(candidate: CodeCandidate, cache: Dict[str, float] | None = None) -> float:
        cached = cache.get(candidate.uri) if cache else None
        if cached is not None:
            score_map[candidate.uri] = cached
            return cached
        score = compute_score(
            tokenizer,
            model,
            query,
            candidate,
            max_length=max_length,
            device=device,
            doc_penalty=doc_penalty,
        )
        score_map[candidate.uri] = score
        if cache is not None:
            cache[candidate.uri] = score
        return score

    for root in roots:
        refresh_score(root)

    def path_score(path: List[CodeCandidate]) -> float:
        if not path:
            return float("-inf")
        total = 0.0
        for node in path:
            total += score_map.get(node.uri, refresh_score(node))
        return total

    active_paths: List[Tuple[List[CodeCandidate], float]] = [([root], path_score([root])) for root in roots]
    active_paths.sort(key=lambda item: item[1], reverse=True)
    active_paths = active_paths[:beam_width]
    best_path = active_paths[0] if active_paths else ([], float("-inf"))
    best_full_path: Tuple[List[CodeCandidate], float] = ([], float("-inf"))
    level_summaries: List[List[Tuple[CodeCandidate, float]]] = []

    for _step in range(1, steps + 1):
        expanded: List[Tuple[List[CodeCandidate], float]] = []
        step_cache: Dict[str, float] = {}
        for path, _score in active_paths:
            if not path:
                continue
            for node in path:
                refresh_score(node, step_cache)
            last = path[-1]
            for neighbor_uri in graph.adjacency.get(last.uri, []):
                neighbor = graph.by_uri.get(neighbor_uri)
                if neighbor is None:
                    continue
                if any(node.uri == neighbor.uri for node in path):
                    continue
                refresh_score(neighbor, step_cache)
                new_path = path + [neighbor]
                expanded.append((new_path, path_score(new_path)))

        if not expanded:
            break

        expanded.sort(key=lambda item: item[1], reverse=True)
        active_paths = expanded[: beam_width or 1]
        candidate_best = active_paths[0]
        if candidate_best[1] > best_path[1]:
            best_path = candidate_best
        for path, score in expanded:
            if len(path) == steps + 1 and score > best_full_path[1]:
                best_full_path = (path, score)

        seen: Dict[str, float] = {}
        level: List[Tuple[CodeCandidate, float]] = []
        for path, _score in expanded:
            node = path[-1]
            if node.uri in seen:
                continue
            node_score = score_map.get(node.uri, float("-inf"))
            seen[node.uri] = node_score
            level.append((node, node_score))
            if len(level) >= (beam_width or 1):
                break
        level_summaries.append(level)

    chosen_path = best_full_path[0] if best_full_path[0] else best_path[0]

    return chosen_path, level_summaries


def main() -> None:
    args = parse_args()
    graph_path = Path(args.graph)
    if not graph_path.exists():
        raise FileNotFoundError(f"Knowledge graph not found: {graph_path}")

    adapter_path = "artifacts/weighting-dora"
    # if not adapter_path.exists():
    #     raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    device = resolve_device(args.device)
    code_graph = load_candidates(graph_path, Path(adapter_path), args.base_model, device)
    candidates = code_graph.candidates
    if not candidates:
        raise RuntimeError("No code candidates were loaded from the graph")

    tokenizer, model = load_model(Path(adapter_path), args.base_model)
    model.to(device)

    ranked, score_map = score_candidates(
        tokenizer,
        model,
        args.query,
        candidates,
        max_length=args.max_length,
        device=device,
        doc_penalty=args.doc_penalty,
    )

    top_results = ranked[: args.top_k]

    def print_results(step: int, results: List[Tuple[CodeCandidate, float]]) -> None:
        print("#" * 80)
        print(f"Step {step} top {len(results)} results")
        for candidate, score in results:
            print("=" * 80)
            print(f"Score: {score:.4f}")
            print(f"File: {candidate.file_path}")
            print(f"Block: {candidate.name} ({candidate.kind})")
            print(candidate.as_passage())

    print_results(0, top_results)

    if args.steps <= 0 or not top_results:
        return

    root_candidates = [candidate for candidate, _ in top_results]
    best_path_nodes, level_summaries = beam_search_paths(
        code_graph,
        root_candidates,
        score_map,
        args.steps,
        args.top_p,
        tokenizer=tokenizer,
        model=model,
        query=args.query,
        max_length=args.max_length,
        device=device,
        doc_penalty=args.doc_penalty,
    )

    for idx, level in enumerate(level_summaries, start=1):
        print_results(idx, level)

    print("#" * 80)
    print("Suggested path through the graph:")
    for index, candidate in enumerate(best_path_nodes):
        score = score_map.get(candidate.uri, float("nan"))
        print(f"Step {index}: {candidate.name} ({candidate.kind}) | Score {score:.4f} | File {candidate.file_path}")

    # Log query if enabled
    if args.enable_logging:
        from src.service.query_log import QueryLogger, get_session_id
        logger = QueryLogger(Path(args.log_path), batch_size=100)
        result_uris = [cand.uri for cand, _ in top_results]
        session_id = get_session_id()
        logger.log_query(args.query, result_uris, session_id, top_k=args.top_k)


if __name__ == "__main__":
    main()
