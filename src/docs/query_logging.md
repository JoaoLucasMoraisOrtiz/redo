# Query Logging and Co-Occurrence Edge Discovery

## Overview

The system automatically tracks query patterns and discovers relationships between code blocks based on co-occurrence in query results. When users search for code, the system logs which nodes appear together and suggests edges to strengthen the knowledge graph.

## How It Works

### 1. Query Logging

When you run retrieval queries with `--enable-logging`, the system records:
- The query text
- Top-K result URIs
- Timestamp
- Session ID (groups related queries)

Logs are stored in `logs/query_log.jsonl` (JSONL format, one record per line).

### 2. Co-Occurrence Analysis

After collecting 100+ queries, the system analyzes patterns:
- **Within-session co-occurrence**: If node A appears in query N and node B appears in query N+1 (same session), they likely form a learning path
- **Sliding window**: Checks up to 5 consecutive queries for co-occurrence
- **Confidence scoring**: Frequency of co-occurrence normalized by total pairs

### 3. Edge Suggestions

The system suggests edges when:
- Two nodes co-occur at least 3 times
- Confidence score > 0.05
- Nodes are different (no self-loops)

Suggested edges are ranked by confidence.

### 4. Graph Enrichment

Edges can be automatically added to the RDF graph with:
- `tw:relatedTo` predicate (directional relationship)
- `tw:coOccurrenceWeight` property (confidence score)

## Usage

### Enable Query Logging

Add `--enable-logging` to your retrieval queries:

```bash
python -m src.service.retrieval "derivative function" --enable-logging
python -m src.service.retrieval "integration method" --enable-logging
python -m src.service.retrieval "arithmetic operations" --enable-logging
# ... run 100+ queries
```

### Analyze Co-Occurrence Patterns

View suggested edges without modifying the graph:

```bash
python -m src.service.analyze_queries
```

### Apply Edges to Graph

Add edges with confidence >= 0.1:

```bash
python -m src.service.analyze_queries --apply --min-confidence 0.1
```

Archive the log after processing:

```bash
python -m src.service.analyze_queries --apply --clear-log
```

### Custom Configuration

```bash
python -m src.service.analyze_queries \
  --log-path logs/custom_log.jsonl \
  --graph src/train_weighting/dataset/my_graph.ttl \
  --window 3 \
  --min-confidence 0.15 \
  --apply
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable-logging` | False | Enable query logging in retrieval CLI |
| `--log-path` | `logs/query_log.jsonl` | Path to query log file |
| `--window` | 5 | Number of consecutive queries to check |
| `--min-confidence` | 0.1 | Minimum confidence for adding edges |
| `--apply` | False | Apply edges to graph (vs. preview only) |
| `--clear-log` | False | Archive log after processing |

## Session Management

Queries in the same terminal session are grouped together. To start a new session:

```bash
unset QUERY_SESSION_ID  # Linux/Mac
$env:QUERY_SESSION_ID = $null  # PowerShell
```

Or explicitly set a session ID:

```bash
export QUERY_SESSION_ID="user-study-01"  # Linux/Mac
$env:QUERY_SESSION_ID = "user-study-01"  # PowerShell
```

## Example Workflow

```bash
# 1. Collect query patterns (run 100+ queries over time)
python -m src.service.retrieval "arithmetic" --enable-logging
python -m src.service.retrieval "division" --enable-logging
python -m src.service.retrieval "remainder" --enable-logging
# ... more queries

# 2. Analyze patterns (automatic after 100 records, or manual)
python -m src.service.analyze_queries

# Output:
# Co-occurrence Analysis Summary:
#   Total sessions: 15
#   Total queries: 120
#   Total co-occurrence pairs: 450
#   Unique edges suggested: 23
#
# Top 10 suggested edges:
#   1. IntegerDivision -> BasicArithmetic (confidence: 0.2150)
#   2. SingleVariableDerivative -> NumericalIntegral (confidence: 0.1820)
#   ...

# 3. Apply edges to graph
python -m src.service.analyze_queries --apply --min-confidence 0.15 --clear-log

# Output:
# ✓ Added 12 co-occurrence edges to the graph
# ✓ Log archived to logs/query_log.processed
```

## Benefits

1. **Automatic Discovery**: No manual edge curation needed
2. **User-Driven**: Edges reflect actual usage patterns
3. **Adaptive**: Graph improves over time with more queries
4. **Non-Intrusive**: Logging is opt-in and doesn't slow queries
5. **Transparent**: All suggestions are reviewable before applying

## Implementation Details

### Log Format

Each line in `query_log.jsonl`:

```json
{
  "timestamp": 1699228800.123,
  "query": "integer division",
  "session_id": "a1b2c3d4",
  "result_uris": [
    "http://example.org/train/codeblock/.../IntegerDivision",
    "http://example.org/train/codeblock/.../BasicArithmetic"
  ],
  "top_k": 5
}
```

### Confidence Calculation

```
confidence = (co_occurrence_count) / (total_pairs_in_window)
```

Higher confidence = more frequent co-occurrence = stronger relationship

### Edge Properties

Added edges include:
- `tw:relatedTo`: Standard directed edge
- `tw:coOccurrenceWeight`: Confidence score (0.0-1.0)

This distinguishes co-occurrence edges from semantic edges (which use `tw:weight`).

## Future Enhancements

- **Decay factor**: Recent co-occurrences weighted higher
- **Bidirectional edges**: Add reverse edges for symmetric relationships
- **Edge pruning**: Remove low-confidence edges after X days
- **A/B testing**: Compare retrieval quality with/without co-occurrence edges
