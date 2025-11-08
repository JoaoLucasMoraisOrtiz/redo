"""Query logging and co-occurrence tracking for automatic edge discovery."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter


@dataclass
class QueryRecord:
    """Record of a single query and its results."""
    timestamp: float
    query: str
    session_id: str
    result_uris: List[str]
    top_k: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> QueryRecord:
        return cls(**data)


class QueryLogger:
    """Tracks query patterns and discovers co-occurrence relationships."""

    def __init__(self, log_path: Path, batch_size: int = 100, co_occurrence_window: int = 5):
        """
        Initialize the query logger.

        Args:
            log_path: Path to the JSONL log file
            batch_size: Number of records before triggering co-occurrence analysis
            co_occurrence_window: Number of consecutive queries to consider for co-occurrence
        """
        self.log_path = log_path
        self.batch_size = batch_size
        self.co_occurrence_window = co_occurrence_window
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_query(self, query: str, result_uris: List[str], session_id: str, top_k: int = 5) -> None:
        """
        Log a query and its results.

        Args:
            query: The search query
            result_uris: List of URIs returned for this query
            session_id: Session identifier to group related queries
            top_k: Number of top results returned
        """
        record = QueryRecord(
            timestamp=time.time(),
            query=query,
            session_id=session_id,
            result_uris=result_uris,
            top_k=top_k,
        )

        # Append to log file
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

        # Check if we should trigger analysis
        record_count = self._count_records()
        if record_count >= self.batch_size:
            print(f"Reached {record_count} records. Triggering co-occurrence analysis...")
            self.analyze_and_suggest_edges()

    def _count_records(self) -> int:
        """Count total records in the log file."""
        if not self.log_path.exists():
            return 0
        with open(self.log_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def load_records(self) -> List[QueryRecord]:
        """Load all query records from the log file."""
        if not self.log_path.exists():
            return []

        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        records.append(QueryRecord.from_dict(data))
                    except (json.JSONDecodeError, TypeError):
                        continue
        return records

    def analyze_and_suggest_edges(self) -> List[Tuple[str, str, float]]:
        """
        Analyze query logs and suggest edges based on co-occurrence patterns.

        Returns:
            List of (source_uri, target_uri, confidence_score) tuples
        """
        records = self.load_records()
        if len(records) < 2:
            return []

        # Group records by session
        sessions: Dict[str, List[QueryRecord]] = defaultdict(list)
        for record in records:
            sessions[record.session_id].append(record)

        # Sort each session by timestamp
        for session_id in sessions:
            sessions[session_id].sort(key=lambda r: r.timestamp)

        # Track co-occurrences within sliding window
        co_occurrences: Counter = Counter()
        total_pairs = 0

        for session_id, session_records in sessions.items():
            for i in range(len(session_records)):
                # Get results from current query
                source_uris = set(session_records[i].result_uris)

                # Look at next queries within the window
                window_end = min(i + self.co_occurrence_window + 1, len(session_records))
                for j in range(i + 1, window_end):
                    target_uris = set(session_records[j].result_uris)

                    # Record co-occurrence pairs
                    for source_uri in source_uris:
                        for target_uri in target_uris:
                            if source_uri != target_uri:
                                co_occurrences[(source_uri, target_uri)] += 1
                                total_pairs += 1

        # Calculate confidence scores and filter
        suggestions = []
        for (source_uri, target_uri), count in co_occurrences.items():
            # Confidence: co-occurrence frequency normalized by total pairs
            confidence = count / total_pairs if total_pairs > 0 else 0.0

            # Only suggest if seen at least 3 times and confidence > 0.05
            if count >= 3 and confidence > 0.05:
                suggestions.append((source_uri, target_uri, confidence))

        # Sort by confidence descending
        suggestions.sort(key=lambda x: x[2], reverse=True)

        # Print summary
        print(f"\nCo-occurrence Analysis Summary:")
        print(f"  Total sessions: {len(sessions)}")
        print(f"  Total queries: {len(records)}")
        print(f"  Total co-occurrence pairs: {total_pairs}")
        print(f"  Unique edges suggested: {len(suggestions)}")
        print()

        if suggestions:
            print("Top 10 suggested edges:")
            for idx, (source, target, conf) in enumerate(suggestions[:10], start=1):
                source_name = source.split("/")[-1][:40]
                target_name = target.split("/")[-1][:40]
                print(f"  {idx}. {source_name} -> {target_name} (confidence: {conf:.4f})")
            print()

        return suggestions

    def apply_suggestions_to_graph(
        self,
        suggestions: List[Tuple[str, str, float]],
        graph_path: Path,
        min_confidence: float = 0.1,
    ) -> int:
        """
        Apply suggested edges to the RDF graph.

        Args:
            suggestions: List of (source_uri, target_uri, confidence) tuples
            graph_path: Path to the RDF graph file
            min_confidence: Minimum confidence threshold for adding edges

        Returns:
            Number of edges added
        """
        from rdflib import Graph, Namespace, URIRef, Literal, XSD

        graph = Graph().parse(graph_path)
        TW = Namespace("http://example.org/train/vocab/")

        edges_added = 0
        for source_uri, target_uri, confidence in suggestions:
            if confidence < min_confidence:
                continue

            source = URIRef(source_uri)
            target = URIRef(target_uri)

            # Check if edge already exists
            if (source, TW.relatedTo, target) in graph:
                continue

            # Add the edge with confidence as weight
            graph.add((source, TW.relatedTo, target))
            graph.add((source, TW.coOccurrenceWeight, Literal(confidence, datatype=XSD.double)))
            edges_added += 1

        if edges_added > 0:
            graph.serialize(destination=str(graph_path), format="turtle")
            print(f"✓ Added {edges_added} co-occurrence edges to the graph")

        return edges_added

    def clear_log(self) -> None:
        """Clear the query log after processing."""
        if self.log_path.exists():
            backup_path = self.log_path.with_suffix(".processed")
            self.log_path.rename(backup_path)
            print(f"✓ Log archived to {backup_path}")


def get_session_id() -> str:
    """Generate or retrieve a session ID for grouping related queries."""
    import os
    import uuid

    # Try to get from environment variable first
    session_id = os.environ.get("QUERY_SESSION_ID")
    if not session_id:
        # Generate a new session ID
        session_id = str(uuid.uuid4())[:8]
        os.environ["QUERY_SESSION_ID"] = session_id
    return session_id
