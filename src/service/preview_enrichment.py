"""CLI tool to preview how a new code snippet would connect into the knowledge graph."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.service.retrieval import (
    CodeCandidate,
    build_embeddings,
    load_candidates,
    load_model,
    preview_enrichment,
    assign_cluster,
    resolve_device,
)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "new-node"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview where a new code snippet would connect inside the knowledge graph",
    )
    parser.add_argument(
        "--code-file",
        type=Path,
        required=True,
        help="Path to a file containing the code snippet to register",
    )
    parser.add_argument(
        "--description",
        required=True,
        help="Short natural language description of what the code does",
    )
    parser.add_argument(
        "--name",
        help="Identifier for the new code block (defaults to a slug from the description)",
    )
    parser.add_argument(
        "--kind",
        default="function",
        help="Kind of the code block (function, method, class, etc.)",
    )
    parser.add_argument(
        "--language",
        default="Java",
        help="Language label stored in the KG",
    )
    parser.add_argument(
        "--version",
        default="java_derivatives v1",
        help="Version label stored in the KG",
    )
    parser.add_argument(
        "--file-path",
        default="java_derivatives/v1/src/main/java/com/example/derivcalc/IntegerDivision.java",
        help="Source file path metadata for the new block",
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
        "--doc-penalty",
        type=float,
        default=1.5,
        help="Penalty applied to documentation-only blocks when scoring",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of candidate connections to preview",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    code_path = args.code_file
    if not code_path.exists():
        raise FileNotFoundError(f"Code file not found: {code_path}")
    code = code_path.read_text(encoding="utf-8")
    if not code.strip():
        raise ValueError("The provided code file is empty")

    description = args.description.strip()
    if not description:
        raise ValueError("Description cannot be empty")

    name = args.name or description
    slug = slugify(name)

    graph_path = Path(args.graph)
    adapter_path = Path(args.adapter)
    device = resolve_device(args.device)

    code_graph = load_candidates(graph_path, adapter_path, args.base_model, device)
    tokenizer, model = load_model(adapter_path, args.base_model)
    model.to(device)

    temp_uri = f"http://example.org/train/temp/{slug}"
    file_uri = f"http://example.org/train/temp/file/{slug}"
    new_candidate = CodeCandidate(
        uri=temp_uri,
        file_uri=file_uri,
        file_path=args.file_path,
        name=name,
        kind=args.kind,
        language=args.language,
        version=args.version,
        code=code,
        description=description,
    )

    suggestions = preview_enrichment(
        code_graph,
        tokenizer,
        model,
        device,
        new_candidate,
        max_length=args.max_length,
        doc_penalty=args.doc_penalty,
        top_k=args.top_k,
    )

    assigned_cluster = None
    if code_graph.cluster_centroids:
        new_embedding = build_embeddings([new_candidate], tokenizer, model, device)[0]
        assigned_cluster = assign_cluster(code_graph.cluster_centroids, new_embedding)

    print("#" * 80)
    print("Preview connections for the new node")
    print(f"Name: {new_candidate.name}")
    print(f"File path: {new_candidate.file_path}")
    print(f"Kind: {new_candidate.kind} | Language: {new_candidate.language} | Version: {new_candidate.version}")
    print(f"Description: {new_candidate.description}")
    if assigned_cluster is not None:
        print(f"Assigned cluster: {assigned_cluster}")
    print("#" * 80)
    if not suggestions:
        print("No candidate connections were suggested.")
        return

    for idx, (candidate, probability) in enumerate(suggestions, start=1):
        print(f"Suggestion {idx}")
        print(f"Probability: {probability:.4f}")
        print(f"URI: {candidate.uri}")
        print(f"File: {candidate.file_path}")
        print(f"Block: {candidate.name} ({candidate.kind})")
        print("-" * 80)


if __name__ == "__main__":
    main()
