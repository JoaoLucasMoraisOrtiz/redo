"""CLI tool to analyze query logs and apply co-occurrence edges to the graph."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.service.query_log import QueryLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze query logs and add co-occurrence edges to the knowledge graph",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/query_log.jsonl"),
        help="Path to the query log file",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("src/train_weighting/dataset/new_graph.ttl"),
        help="Path to the RDF/Turtle knowledge graph",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.1,
        help="Minimum confidence threshold for adding edges (0.0-1.0)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Number of consecutive queries to consider for co-occurrence",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply suggested edges to the graph (default: only show suggestions)",
    )
    parser.add_argument(
        "--clear-log",
        action="store_true",
        help="Archive and clear the log file after processing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.log_path.exists():
        print(f"No log file found at {args.log_path}")
        print("Run queries with --enable-logging to collect data first.")
        return

    logger = QueryLogger(
        log_path=args.log_path,
        batch_size=100,
        co_occurrence_window=args.window,
    )

    # Analyze and get suggestions
    suggestions = logger.analyze_and_suggest_edges()

    if not suggestions:
        print("No co-occurrence patterns found meeting the criteria.")
        return

    # Apply to graph if requested
    if args.apply:
        if not args.graph.exists():
            print(f"Graph file not found: {args.graph}")
            return

        print(f"\nApplying edges with confidence >= {args.min_confidence} to {args.graph}...")
        edges_added = logger.apply_suggestions_to_graph(
            suggestions,
            args.graph,
            min_confidence=args.min_confidence,
        )

        if edges_added > 0:
            print(f"✓ Successfully added {edges_added} edges")

            if args.clear_log:
                logger.clear_log()
        else:
            print("No edges met the confidence threshold.")
    else:
        print("\nTo apply these edges to the graph, run with --apply flag")
        print(f"Example: python -m src.service.analyze_queries --apply --min-confidence {args.min_confidence}")


if __name__ == "__main__":
    main()
