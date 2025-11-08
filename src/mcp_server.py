"""MCP Server for Code Knowledge Graph Operations."""
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from pathlib import Path
import subprocess
import json
import sys
import os
from typing import Any
from datetime import datetime


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Statistics tracking
stats_file = project_root / "src" / "service" / "search_statistics.jsonl"
STATS_THRESHOLD = 100


app = Server("code-kg-server")


def log_search_statistic(query_type: str, query_value: str, result_nodes: list[str]):
    """Log search statistics and trigger co-occurrence analysis if threshold reached."""
    stat = {
        "timestamp": datetime.now().isoformat(),
        "query_type": query_type,
        "query_value": query_value,
        "result_nodes": result_nodes
    }
    
    # Append to JSONL
    with open(stats_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(stat) + "\n")
    
    # Check if we've hit the threshold
    line_count = 0
    if stats_file.exists():
        with open(stats_file, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
    
    if line_count >= STATS_THRESHOLD:
        # Trigger co-occurrence analysis
        subprocess.run(
            [sys.executable, "-m", "src.service.analyze_queries",
             "--apply", "--min-confidence", "0.15", "--clear-log"],
            cwd=str(project_root),
            capture_output=True
        )
        # Archive statistics
        archive_path = stats_file.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        stats_file.rename(archive_path)


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    return [
        Tool(
            name="search_code",
            description="Search the knowledge graph for code snippets using natural language queries. Returns ranked results with semantic scoring.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of the desired code"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return",
                        "default": 5
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of graph traversal steps for beam search",
                        "default": 0
                    },
                    "beam_width": {
                        "type": "integer",
                        "description": "Beam width for graph expansion",
                        "default": 3
                    },
                    "enable_logging": {
                        "type": "boolean",
                        "description": "Enable query logging for co-occurrence analysis",
                        "default": False
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="preview_node_connections",
            description="Preview where a new code snippet would connect in the knowledge graph without actually adding it. Returns suggested connections with probabilities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code snippet content"
                    },
                    "description": {
                        "type": "string",
                        "description": "Natural language description of what the code does"
                    },
                    "name": {
                        "type": "string",
                        "description": "Name/identifier for the code block"
                    },
                    "kind": {
                        "type": "string",
                        "description": "Kind of code block (function, method, class, etc.)",
                        "default": "function"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language",
                        "default": "Java"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of connection suggestions to return",
                        "default": 10
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum probability threshold for suggestions",
                        "default": 0.5
                    }
                },
                "required": ["code", "description", "name"]
            }
        ),
        Tool(
            name="add_node",
            description="Add a new code snippet to the knowledge graph with automatic semantic connections. Creates RDF triples and links to related code blocks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code snippet content"
                    },
                    "description": {
                        "type": "string",
                        "description": "Natural language description of what the code does"
                    },
                    "name": {
                        "type": "string",
                        "description": "Name/identifier for the code block"
                    },
                    "kind": {
                        "type": "string",
                        "description": "Kind of code block (function, method, class, etc.)",
                        "default": "function"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language",
                        "default": "Java"
                    },
                    "version": {
                        "type": "string",
                        "description": "Project version label",
                        "default": "java_derivatives v1"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Source file path for metadata",
                        "default": "java_derivatives/v1/src/main/java/com/example/derivcalc/NewCode.java"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum probability threshold for creating connections",
                        "default": 0.9
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of connections to create",
                        "default": 5
                    }
                },
                "required": ["code", "description", "name"]
            }
        ),
        Tool(
            name="analyze_query_patterns",
            description="Analyze query logs to discover co-occurrence patterns and suggest new edges based on usage. Returns suggested edges with confidence scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "apply": {
                        "type": "boolean",
                        "description": "Apply suggested edges to the graph (vs preview only)",
                        "default": False
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence threshold for adding edges",
                        "default": 0.1
                    },
                    "window": {
                        "type": "integer",
                        "description": "Number of consecutive queries to consider for co-occurrence",
                        "default": 5
                    },
                    "clear_log": {
                        "type": "boolean",
                        "description": "Archive and clear the log after processing",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_node_details",
            description="Retrieve detailed information about a specific code block by its URI or name. Returns code, metadata, and connections.",
            inputSchema={
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "Node URI or name to look up"
                    }
                },
                "required": ["identifier"]
            }
        ),
        Tool(
            name="list_graph_statistics",
            description="Get statistics about the knowledge graph: total nodes, edges, languages, versions, etc.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="search_by_name",
            description="Search for code blocks by exact or partial name match. Returns nodes whose name contains the search term.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name or partial name to search for"
                    },
                    "exact_match": {
                        "type": "boolean",
                        "description": "Require exact name match (case-insensitive)",
                        "default": False
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 10
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="search_by_vector",
            description="Pure vector similarity search using embeddings (RAG-style, no graph traversal). Fast semantic search without graph structure.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description for semantic search"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of most similar results to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_by_code_term",
            description="Search for occurrences of a specific term (variable, function, class name) in code blocks. Returns all nodes containing that term.",
            inputSchema={
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "Code term to search for (function/variable/class name)"
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Case-sensitive search",
                        "default": True
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 20
                    }
                },
async def search_code(args: dict) -> list[TextContent]:
    """Execute code search."""
    cmd = [
        sys.executable, "-m", "src.service.retrieval",
        args["query"],
        "--top-k", str(args.get("top_k", 5)),
        "--steps", str(args.get("steps", 0)),
        "--top-p", str(args.get("beam_width", 3))
    ]
    
    if args.get("enable_logging", False):
        cmd.append("--enable-logging")
    
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True
    )
    
    # Extract result node names for statistics (parse from stdout)
    result_nodes = []
    for line in result.stdout.split("\n"):
        if "Name:" in line:
            name = line.split("Name:")[1].strip()
            result_nodes.append(name)
    
    log_search_statistic("semantic_search", args["query"], result_nodes)
    
    return [TextContent(
        type="text",
        text=f"Search Results:\n\n{result.stdout}\n\nErrors (if any):\n{result.stderr}"
    )]  return await search_by_name(arguments)
    elif name == "search_by_vector":
        return await search_by_vector(arguments)
    elif name == "search_by_code_term":
        return await search_by_code_term(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def search_code(args: dict) -> list[TextContent]:
    """Execute code search."""
    cmd = [
        sys.executable, "-m", "src.service.retrieval",
        args["query"],
        "--top-k", str(args.get("top_k", 5)),
        "--steps", str(args.get("steps", 0)),
        "--top-p", str(args.get("beam_width", 3))
    ]
    
    if args.get("enable_logging", False):
        cmd.append("--enable-logging")
    
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True
    )
    
    return [TextContent(
        type="text",
        text=f"Search Results:\n\n{result.stdout}\n\nErrors (if any):\n{result.stderr}"
    )]


async def preview_node_connections(args: dict) -> list[TextContent]:
    """Preview node connections."""
    # Create temporary file with code
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
        f.write(args["code"])
        temp_path = f.name
    
    try:
        cmd = [
            sys.executable, "-m", "src.service.preview_enrichment",
            "--code-file", temp_path,
            "--description", args["description"],
            "--name", args["name"],
            "--kind", args.get("kind", "function"),
            "--language", args.get("language", "Java"),
            "--top-k", str(args.get("top_k", 10)),
            "--threshold", str(args.get("threshold", 0.5))
        ]
        
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        
        return [TextContent(
            type="text",
            text=f"Connection Preview:\n\n{result.stdout}\n\nErrors (if any):\n{result.stderr}"
        )]
    finally:
        os.unlink(temp_path)


async def add_node(args: dict) -> list[TextContent]:
    """Add a node to the graph."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
        f.write(args["code"])
        temp_path = f.name
    
    try:
        cmd = [
            sys.executable, "-m", "src.service.add_node",
            "--code-file", temp_path,
            "--description", args["description"],
            "--name", args["name"],
            "--kind", args.get("kind", "function"),
            "--language", args.get("language", "Java"),
            "--version", args.get("version", "java_derivatives v1"),
            "--file-path", args.get("file_path", "java_derivatives/v1/src/main/java/com/example/derivcalc/NewCode.java"),
            "--threshold", str(args.get("threshold", 0.9)),
            "--top-k", str(args.get("top_k", 5))
        ]
        
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        
        return [TextContent(
            type="text",
            text=f"Add Node Result:\n\n{result.stdout}\n\nErrors (if any):\n{result.stderr}"
        )]
    finally:
        os.unlink(temp_path)


async def analyze_query_patterns(args: dict) -> list[TextContent]:
    """Analyze query patterns and suggest edges."""
    cmd = [
        sys.executable, "-m", "src.service.analyze_queries",
        "--window", str(args.get("window", 5)),
        "--min-confidence", str(args.get("min_confidence", 0.1))
    ]
    
    if args.get("apply", False):
        cmd.append("--apply")
    
    if args.get("clear_log", False):
        cmd.append("--clear-log")
    
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True
    )
    
    return [TextContent(
        type="text",
        text=f"Query Analysis:\n\n{result.stdout}\n\nErrors (if any):\n{result.stderr}"
    )]


async def get_node_details(args: dict) -> list[TextContent]:
    """Get details about a specific node."""
    from rdflib import Graph, Namespace
    
    graph_path = project_root / "src" / "train_weighting" / "dataset" / "new_graph.ttl"
    graph = Graph().parse(graph_path)
    TW = Namespace("http://example.org/train/vocab/")
    
    identifier = args["identifier"]
    
    # Try to find node by URI or name
    results = []
    for s, p, o in graph:
        if identifier in str(s) or (p == TW.name and identifier.lower() in str(o).lower()):
            # Get all properties of this subject
            props = {}
            for pred, obj in graph.predicate_objects(s):
                pred_name = str(pred).split("/")[-1]
                props[pred_name] = str(obj)
            
            results.append({
                "uri": str(s),
                "properties": props
            })
    
    if not results:
        return [TextContent(
            type="text",
            text=f"No node found matching identifier: {identifier}"
        )]
    
    output = f"Found {len(results)} node(s):\n\n"
    for idx, node in enumerate(results[:5], 1):
        output += f"Node {idx}:\n"
        output += f"  URI: {node['uri']}\n"
        for key, value in node['properties'].items():
            if len(value) > 100:
                value = value[:100] + "..."
            output += f"  {key}: {value}\n"
        output += "\n"
    
    return [TextContent(type="text", text=output)]


async def list_graph_statistics(args: dict) -> list[TextContent]:
    """Get graph statistics."""
    from rdflib import Graph, Namespace, RDF
    
    graph_path = project_root / "src" / "train_weighting" / "dataset" / "new_graph.ttl"
    graph = Graph().parse(graph_path)
    TW = Namespace("http://example.org/train/vocab/")
    
    # Count nodes, edges, languages, etc.
    total_nodes = len(list(graph.subjects(RDF.type, TW.CodeBlock)))
    total_edges = len(list(graph.triples((None, TW.relatedTo, None))))
    languages = set()
    versions = set()
    kinds = {}
    
    for block in graph.subjects(RDF.type, TW.CodeBlock):
        lang_uri = graph.value(block, TW.belongsToLanguage)
        if lang_uri:
            lang_label = graph.value(lang_uri, TW.label)
            if lang_label:
                languages.add(str(lang_label))
        
        ver_uri = graph.value(block, TW.belongsToVersion)
        if ver_uri:
            ver_label = graph.value(ver_uri, TW.label)
            if ver_label:
                versions.add(str(ver_label))
        
        kind = graph.value(block, TW.kind)
        if kind:
            kind_str = str(kind)
            kinds[kind_str] = kinds.get(kind_str, 0) + 1
    
    output = f"""Knowledge Graph Statistics:

Total Code Blocks: {total_nodes}
    return [TextContent(type="text", text=output)]


async def search_by_name(args: dict) -> list[TextContent]:
    """Search for nodes by name."""
    from rdflib import Graph, Namespace
    
    graph_path = project_root / "src" / "train_weighting" / "dataset" / "new_graph.ttl"
    graph = Graph().parse(graph_path)
    TW = Namespace("http://example.org/train/vocab/")
    
    search_name = args["name"]
    exact_match = args.get("exact_match", False)
    limit = args.get("limit", 10)
    
    results = []
    for block in graph.subjects():
        name = graph.value(block, TW.name)
        if name:
            name_str = str(name)
            if exact_match:
                if name_str.lower() == search_name.lower():
                    results.append((str(block), name_str))
            else:
                if search_name.lower() in name_str.lower():
                    results.append((str(block), name_str))
        
        if len(results) >= limit:
            break
    
    result_nodes = [name for _, name in results]
    log_search_statistic("name_search", search_name, result_nodes)
    
    if not results:
        return [TextContent(
            type="text",
            text=f"No nodes found matching name: {search_name}"
        )]
    
    output = f"Found {len(results)} node(s) matching '{search_name}':\n\n"
    for idx, (uri, name) in enumerate(results, 1):
        # Get description
        desc = graph.value(uri, TW.description)
        kind = graph.value(uri, TW.kind)
        
        output += f"{idx}. {name}\n"
        output += f"   URI: {uri}\n"
        if kind:
            output += f"   Kind: {kind}\n"
        if desc:
            desc_str = str(desc)[:150] + "..." if len(str(desc)) > 150 else str(desc)
            output += f"   Description: {desc_str}\n"
        output += "\n"
    
    return [TextContent(type="text", text=output)]


async def search_by_vector(args: dict) -> list[TextContent]:
    """Pure vector similarity search (RAG-style)."""
    from rdflib import Graph, Namespace, RDF
    from transformers import AutoTokenizer, AutoModel
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    graph_path = project_root / "src" / "train_weighting" / "dataset" / "new_graph.ttl"
    graph = Graph().parse(graph_path)
    TW = Namespace("http://example.org/train/vocab/")
    
    # Load model for embeddings
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    def get_embedding(text: str) -> np.ndarray:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use CLS token embedding
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
    query = args["query"]
    top_k = args.get("top_k", 10)
    
    # Get query embedding
    query_emb = get_embedding(query)
    
    # Get all code blocks and their embeddings
    candidates = []
    for block in graph.subjects(RDF.type, TW.CodeBlock):
        name = graph.value(block, TW.name)
        desc = graph.value(block, TW.description)
        code = graph.value(block, TW.code)
        
        if not all([name, desc, code]):
            continue
        
        # Create text for embedding
        text = f"{name} {desc} {code}"
        emb = get_embedding(text)
        
        candidates.append({
            "uri": str(block),
            "name": str(name),
            "description": str(desc),
            "embedding": emb
        })
    
    # Compute similarities
    embeddings = np.stack([c["embedding"] for c in candidates])
    similarities = cosine_similarity([query_emb], embeddings)[0]
    
    # Sort by similarity
    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in ranked_indices:
        results.append({
            "name": candidates[idx]["name"],
            "uri": candidates[idx]["uri"],
            "description": candidates[idx]["description"],
            "similarity": float(similarities[idx])
        })
    
    result_nodes = [r["name"] for r in results]
    log_search_statistic("vector_search", query, result_nodes)
    
    output = f"Vector Similarity Search Results for: '{query}'\n\n"
    for idx, result in enumerate(results, 1):
        output += f"{idx}. {result['name']} (similarity: {result['similarity']:.3f})\n"
        desc = result['description'][:150] + "..." if len(result['description']) > 150 else result['description']
        output += f"   {desc}\n"
        output += f"   URI: {result['uri']}\n\n"
    
    return [TextContent(type="text", text=output)]


async def search_by_code_term(args: dict) -> list[TextContent]:
    """Search for code term occurrences."""
    from rdflib import Graph, Namespace, RDF
    import re
    
    graph_path = project_root / "src" / "train_weighting" / "dataset" / "new_graph.ttl"
    graph = Graph().parse(graph_path)
    TW = Namespace("http://example.org/train/vocab/")
    
    term = args["term"]
    case_sensitive = args.get("case_sensitive", True)
    limit = args.get("limit", 20)
    
    # Create regex pattern
    if case_sensitive:
        pattern = re.compile(r'\b' + re.escape(term) + r'\b')
    else:
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
    
    results = []
    for block in graph.subjects(RDF.type, TW.CodeBlock):
        code = graph.value(block, TW.code)
        if not code:
            continue
        
        code_str = str(code)
        matches = pattern.findall(code_str)
        
        if matches:
            name = graph.value(block, TW.name)
            kind = graph.value(block, TW.kind)
            
            # Find line numbers where term appears
            lines_with_term = []
            for line_num, line in enumerate(code_str.split("\n"), 1):
                if pattern.search(line):
                    lines_with_term.append((line_num, line.strip()))
            
            results.append({
                "uri": str(block),
                "name": str(name),
                "kind": str(kind) if kind else "unknown",
                "occurrences": len(matches),
                "lines": lines_with_term[:5]  # First 5 occurrences
            })
        
        if len(results) >= limit:
            break
    
    result_nodes = [r["name"] for r in results]
    log_search_statistic("code_term_search", term, result_nodes)
    
    if not results:
        return [TextContent(
            type="text",
            text=f"No occurrences found for term: {term}"
        )]
    
    output = f"Found '{term}' in {len(results)} code block(s):\n\n"
    for idx, result in enumerate(results, 1):
        output += f"{idx}. {result['name']} ({result['kind']})\n"
        output += f"   Occurrences: {result['occurrences']}\n"
        output += f"   URI: {result['uri']}\n"
        if result['lines']:
            output += "   Sample lines:\n"
            for line_num, line in result['lines'][:3]:
                line_preview = line[:80] + "..." if len(line) > 80 else line
                output += f"     Line {line_num}: {line_preview}\n"
        output += "\n"
    
    return [TextContent(type="text", text=output)]


async def main():
Code Block Types:
"""
    for kind, count in sorted(kinds.items(), key=lambda x: x[1], reverse=True):
        output += f"  {kind}: {count}\n"
    
    return [TextContent(type="text", text=output)]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
