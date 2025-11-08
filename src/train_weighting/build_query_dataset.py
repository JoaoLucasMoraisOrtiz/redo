"""Generate natural language query/answer pairs from the training graph."""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from rdflib import Graph, Namespace, RDF

BASE_NAMESPACE = Namespace("http://example.org/train/")
TW = Namespace("http://example.org/train/vocab/")


@dataclass
class BlockSample:
    uri: str
    name: str
    kind: str
    code: str
    file_path: str
    language_label: str
    version_label: str

    def intent_tokens(self) -> List[str]:
        tokens = re.split(r"[^A-Za-z0-9]+", self.name)
        return [token.lower() for token in tokens if token]


def clip_text(value: str, max_chars: int = 900) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


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


QUESTION_PATTERNS = [
    "In the {language} project ({version}), which code block handles {intent}?",
    "Which part of the {language} {version} project implements {intent}?",
    "Where can I find the implementation of {intent} in the {language} project ({version})?",
    "Show me the {intent} logic in the {language} {version} project.",
]

FALLBACK_PATTERNS = [
    "Where is the {name} block defined in the {language} project ({version})?",
    "Which file contains the {name} block for the {language} {version} project?",
]


def make_queries(sample: BlockSample, limit: int) -> List[str]:
    intents = sample.intent_tokens()
    if intents:
        intent_phrase = " ".join(intents)
        patterns = QUESTION_PATTERNS
        template_values = {
            "language": sample.language_label,
            "version": sample.version_label,
            "intent": intent_phrase,
        }
    else:
        patterns = FALLBACK_PATTERNS
        template_values = {
            "language": sample.language_label,
            "version": sample.version_label,
            "name": sample.name,
        }
    rendered = [pattern.format(**template_values) for pattern in patterns]
    return rendered[:limit]


def make_answer(sample: BlockSample) -> str:
    header = (
        f"Block {sample.name} ({sample.kind}) located in {sample.file_path} "
        f"for {sample.language_label} / {sample.version_label}."
    )
    return clip_text(f"{header}\n\nCode:\n{sample.code}")


def build_dataset(samples: Iterable[BlockSample], per_block: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    dataset: List[dict] = []
    for sample in shuffled:
        answer = make_answer(sample)
        for query in make_queries(sample, per_block):
            dataset.append({"query": query, "answer": answer})
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create natural-language query dataset from the training graph")
    parser.add_argument(
        "--dataset",
        default="src/train_weighting/dataset/v2/library_training.ttl",
        help="Path to the RDF dataset (Turtle)",
    )
    parser.add_argument(
        "--output",
        default="src/train_weighting/dataset/v2/query_pairs.json",
        help="Where to write the generated JSON",
    )
    parser.add_argument("--per-block", type=int, default=3, help="Max queries to generate per code block")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for shuffling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    samples = load_block_samples(dataset_path)
    if not samples:
        raise RuntimeError("No code block samples were loaded from the dataset")

    pairs = build_dataset(samples, per_block=args.per_block, seed=args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(pairs, handle, ensure_ascii=True, indent=2)
    print(f"Generated {len(pairs)} query/answer pairs at {output_path}")


if __name__ == "__main__":
    main()
