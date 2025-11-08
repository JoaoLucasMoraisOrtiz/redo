"""Initialize a new RDF knowledge graph."""
import argparse
from pathlib import Path
from rdflib import Graph, Namespace, Literal, RDF, RDFS, URIRef


def create_empty_graph(output_path: Path):
    """Create a new empty RDF graph with base ontology."""
    graph = Graph()
    
    # Define namespace
    TW = Namespace("http://example.org/train/vocab/")
    graph.bind("tw", TW)
    graph.bind("rdf", RDF)
    graph.bind("rdfs", RDFS)
    
    # Define ontology classes
    graph.add((TW.CodeBlock, RDF.type, RDFS.Class))
    graph.add((TW.CodeBlock, RDFS.label, Literal("Code Block")))
    graph.add((TW.CodeBlock, RDFS.comment, Literal("A block of source code (class, method, function, etc.)")))
    
    graph.add((TW.Language, RDF.type, RDFS.Class))
    graph.add((TW.Language, RDFS.label, Literal("Programming Language")))
    
    graph.add((TW.Version, RDF.type, RDFS.Class))
    graph.add((TW.Version, RDFS.label, Literal("Project Version")))
    
    # Define properties
    properties = [
        (TW.name, "Name of the code block"),
        (TW.description, "Natural language description"),
        (TW.code, "Source code content"),
        (TW.kind, "Type of code block (class, method, function, etc.)"),
        (TW.language, "Programming language name"),
        (TW.filePath, "Source file path"),
        (TW.belongsToLanguage, "Language this code belongs to"),
        (TW.belongsToVersion, "Version/project this code belongs to"),
        (TW.relatedTo, "Semantic relationship to another code block"),
        (TW.weight, "Edge weight (similarity score)"),
        (TW.label, "Human-readable label"),
        (TW.coOccurrenceWeight, "Co-occurrence confidence from query logs"),
    ]
    
    for prop, comment in properties:
        graph.add((prop, RDF.type, RDF.Property))
        graph.add((prop, RDFS.comment, Literal(comment)))
    
    # Add common languages
    languages = ["Java", "Python", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust"]
    for lang in languages:
        lang_uri = TW[f"language_{lang.lower()}"]
        graph.add((lang_uri, RDF.type, TW.Language))
        graph.add((lang_uri, TW.label, Literal(lang)))
    
    # Save graph
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(output_path), format="turtle")
    
    print(f"✓ Created empty knowledge graph at: {output_path}")
    print(f"  - {len(list(graph.triples((None, RDF.type, RDFS.Class))))} classes defined")
    print(f"  - {len(list(graph.triples((None, RDF.type, RDF.Property))))} properties defined")
    print(f"  - {len(languages)} languages pre-configured")
    print(f"  - Total triples: {len(graph)}")


def main():
    parser = argparse.ArgumentParser(description="Initialize a new RDF knowledge graph")
    parser.add_argument(
        "--output",
        type=str,
        default="src/train_weighting/dataset/new_graph.ttl",
        help="Output path for the new graph file"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing file if it exists"
    )
    
    args = parser.parse_args()
    output_path = Path(args.output)
    
    if output_path.exists() and not args.overwrite:
        print(f"Error: File already exists: {output_path}")
        print("Use --overwrite to replace it")
        return
    
    create_empty_graph(output_path)


if __name__ == "__main__":
    main()
