"""Train the weighting CrossEncoder with a DoRA adapter using contrastive learning."""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from rdflib import Graph, Namespace, RDF

from src.service.training import TripletExample, WeightingModelTrainer

BASE_NAMESPACE = Namespace("http://example.org/train/")
TW = Namespace("http://example.org/train/vocab/")


@dataclass
class BlockSample:
    """Lightweight container for a code block and its metadata."""

    uri: str
    name: str
    kind: str
    code: str
    file_path: str
    language_label: str
    version_label: str

    def intent_phrase(self) -> str:
        tokens = re.split(r"[^A-Za-z0-9]+", self.name)
        readable = " ".join(token.lower() for token in tokens if token)
        return readable or self.name.lower()


def clip_text(value: str, max_chars: int = 900) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def load_query_pairs(json_path: Path) -> List[Dict[str, str]]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Pair dataset must be a list of objects, got {type(data)}")
    pairs: List[Dict[str, str]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        query = entry.get("query")
        answer = entry.get("answer")
        if isinstance(query, str) and isinstance(answer, str):
            pairs.append({"query": query, "answer": answer})
    if not pairs:
        raise ValueError("No valid query/answer pairs were found in the JSON dataset")
    return pairs


def load_block_samples(graph_path: Path) -> List[BlockSample]:
    graph = Graph().parse(graph_path)
    samples: List[BlockSample] = []
    for block_uri in graph.subjects(RDF.type, TW.CodeBlock):
        name_literal = graph.value(block_uri, TW.name)
        code_literal = graph.value(block_uri, TW.code)
        kind_literal = graph.value(block_uri, TW.kind)
        file_uri = graph.value(block_uri, TW.belongsToFile)
        version_uri = graph.value(block_uri, TW.belongsToVersion)
        language_uri = graph.value(block_uri, TW.belongsToLanguage)
        if not all((name_literal, code_literal, file_uri, version_uri, language_uri)):
            continue
        file_path_literal = graph.value(file_uri, TW.path)
        language_label = graph.value(language_uri, TW.label)
        version_label = graph.value(version_uri, TW.label)
        samples.append(
            BlockSample(
                uri=str(block_uri),
                name=str(name_literal),
                kind=str(kind_literal) if kind_literal else "section",
                code=str(code_literal),
                file_path=str(file_path_literal) if file_path_literal else "",
                language_label=str(language_label) if language_label else "",
                version_label=str(version_label) if version_label else "",
            )
        )
    return samples


def make_anchor(sample: BlockSample) -> str:
    intent = sample.intent_phrase()
    if intent:
        return (
            f"In the {sample.language_label} project ({sample.version_label}), "
            f"which code block handles {intent}?"
        )
    return (
        f"Where is the {sample.name} block defined in the "
        f"{sample.language_label} project ({sample.version_label})?"
    )


def make_positive(sample: BlockSample) -> str:
    header = (
        f"Block {sample.name} ({sample.kind}) located in {sample.file_path} "
        f"for {sample.language_label} / {sample.version_label}."
    )
    return clip_text(f"{header}\n\nCode:\n{sample.code}")


def make_negative(sample: BlockSample) -> str:
    header = (
        f"Block {sample.name} ({sample.kind}) located in {sample.file_path} "
        f"for {sample.language_label} / {sample.version_label}."
    )
    return clip_text(f"{header}\n\nCode:\n{sample.code}")


def build_triplets(samples: Iterable[BlockSample], seed: int = 13) -> List[TripletExample]:
    grouped: Dict[str, List[BlockSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.language_label, []).append(sample)
    rng = random.Random(seed)
    triplets: List[TripletExample] = []
    for group in grouped.values():
        if len(group) < 2:
            continue
        for sample in group:
            negatives = [candidate for candidate in group if candidate.uri != sample.uri]
            if not negatives:
                continue
            negative_sample = rng.choice(negatives)
            triplets.append(
                TripletExample(
                    anchor=make_anchor(sample),
                    positive=make_positive(sample),
                    negative=make_negative(negative_sample),
                )
            )
    return triplets


def extract_language_from_answer(answer: str) -> str:
    first_line = answer.splitlines()[0] if answer else ""
    marker = "for "
    if marker in first_line:
        segment = first_line.split(marker, 1)[1]
        language = segment.split("/", 1)[0].strip().lower()
        if language:
            return language
    return "unknown"


def build_triplets_from_pairs(pairs: List[Dict[str, str]], seed: int = 13) -> List[TripletExample]:
    if len(pairs) < 2:
        raise ValueError("Need at least two query/answer pairs to build triplets")
    rng = random.Random(seed)
    per_language: Dict[str, List[int]] = {}
    for idx, pair in enumerate(pairs):
        language = extract_language_from_answer(pair["answer"])
        per_language.setdefault(language, []).append(idx)
    triplets: List[TripletExample] = []
    for idx, pair in enumerate(pairs):
        language = extract_language_from_answer(pair["answer"])
        candidates = [candidate for candidate in per_language.get(language, []) if candidate != idx]
        if not candidates:
            candidates = [candidate for candidate in range(len(pairs)) if candidate != idx]
        negative_index = rng.choice(candidates)
        triplets.append(
            TripletExample(
                anchor=pair["query"],
                positive=pair["answer"],
                negative=pairs[negative_index]["answer"],
            )
        )
    return triplets


def resolve_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
    except ImportError:
        return "cpu"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the weighting model with DoRA and contrastive learning")
    parser.add_argument(
        "--dataset",
        default="src/train_weighting/dataset/v2/library_training.ttl",
        help="Path to the RDF dataset in Turtle format",
    )
    parser.add_argument(
        "--pair-dataset",
        default="src/train_weighting/dataset/v2/query_pairs_manual.json",
        help="Path to JSON file with handcrafted query/answer pairs",
    )
    parser.add_argument(
        "--output",
        default="artifacts/weighting-dora",
        help="Where to store the adapted model",
    )
    parser.add_argument(
        "--model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Base cross-encoder model name",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum token length per input")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Optimizer learning rate")
    parser.add_argument("--device", help="Torch device to use (auto-detected if not provided)")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility")
    parser.add_argument("--lora-r", type=int, default=8, help="Rank for LoRA/DoRA adaptation")
    parser.add_argument("--lora-alpha", type=int, default=16, help="Scaling factor for LoRA/DoRA")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="Dropout for LoRA/DoRA adaptation")
    parser.add_argument(
        "--no-dora",
        action="store_true",
        help="Disable DoRA adaptation (falls back to full fine-tuning if set)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_dataset_path = Path(args.pair_dataset) if args.pair_dataset else None
    triplets: List[TripletExample]

    if pair_dataset_path and pair_dataset_path.exists():
        pairs = load_query_pairs(pair_dataset_path)
        triplets = build_triplets_from_pairs(pairs, seed=args.seed)
    else:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        samples = load_block_samples(dataset_path)
        triplets = build_triplets(samples, seed=args.seed)

    if not triplets:
        raise RuntimeError("No triplets generated from the dataset; check dataset integrity")

    device = resolve_device(args.device)
    trainer = WeightingModelTrainer(
        model_name=args.model,
        lr=args.learning_rate,
        use_dora=not args.no_dora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    dataloader = trainer.build_dataloader(triplets, batch_size=args.batch_size, max_length=args.max_length)
    trainer.train(dataloader, epochs=args.epochs, device=device)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(str(output_dir))
    print(f"Model trained and saved to {output_dir}")


if __name__ == "__main__":
    main()
