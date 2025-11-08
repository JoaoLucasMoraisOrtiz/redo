import argparse
import hashlib
import re
from pathlib import Path
from typing import Dict, List

from rdflib import Graph, Literal, Namespace, RDF, URIRef

BASE_NAMESPACE = Namespace("http://example.org/train/")
TW = Namespace("http://example.org/train/vocab/")


def slugify(value: str) -> str:
    cleaned = ''.join(ch if ch.isalnum() else '-' for ch in value)
    cleaned = cleaned.strip('-') or 'resource'
    return cleaned.lower()


def file_uri(language: str, version: str, relative_path: str) -> URIRef:
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()
    return BASE_NAMESPACE[f"file/{language}/{version}/{digest}"]


def code_block_uri(language: str, version: str, relative_path: str, index: int) -> URIRef:
    token = f"{relative_path}::{index}"
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return BASE_NAMESPACE[f"codeblock/{language}/{version}/{digest}"]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return path.read_text(encoding='latin-1')


def extract_java_blocks(text: str) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []
    class_pattern = re.compile(r'(^\s*(public\s+)?(final\s+)?class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)[^\{]*\{)', re.MULTILINE)
    method_pattern = re.compile(
        r'(^\s*(public|protected|private)?\s*(static\s+)?[\w\<\>\[\]]+\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*(throws[^\{]*)?\{)',
        re.MULTILINE,
    )

    def extract_block(start: int) -> int:
        depth = 0
        i = start
        in_string: str = ''
        while i < len(text):
            char = text[i]
            if in_string:
                if char == in_string and text[i - 1] != '\\':
                    in_string = ''
            else:
                if char in ('"', '\''):
                    in_string = char
                elif char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return i + 1
            i += 1
        return len(text)

    for pattern, kind in ((class_pattern, 'class'), (method_pattern, 'method')):
        for match in pattern.finditer(text):
            start = match.start()
            brace_start = match.end() - 1
            if text[brace_start] != '{':
                continue
            end = extract_block(brace_start)
            code = text[start:end]
            name = match.group('name') if 'name' in match.groupdict() else 'anonymous'
            blocks.append({'name': name, 'code': code, 'kind': kind})
    return blocks if blocks else [{'name': 'file', 'code': text, 'kind': 'file'}]


def extract_cobol_blocks(text: str) -> List[Dict[str, str]]:
    lines = text.splitlines()
    pattern = re.compile(r'^\s{0,7}([A-Z0-9-]+)\.')
    blocks: List[Dict[str, str]] = []
    current_name = None
    current_lines: List[str] = []

    def flush() -> None:
        if current_name and current_lines:
            blocks.append({'name': current_name, 'code': '\n'.join(current_lines), 'kind': 'paragraph'})

    for line in lines:
        match = pattern.match(line)
        if match:
            flush()
            current_name = match.group(1)
            current_lines = [line]
        else:
            if current_lines:
                current_lines.append(line)
    flush()
    return blocks if blocks else [{'name': 'file', 'code': text, 'kind': 'file'}]


def extract_delphi_blocks(text: str) -> List[Dict[str, str]]:
    lines = text.splitlines()
    signature_pattern = re.compile(
        r'^\s*(procedure|function|constructor|destructor)\s+([A-Za-z0-9_.]+)\s*(\([^;]*\))?\s*;?\s*(overload|override|virtual|reintroduce|abstract|stdcall|cdecl|register|dynamic|inline)?',
        re.IGNORECASE,
    )
    blocks: List[Dict[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        match = signature_pattern.match(line)
        if match:
            kind = match.group(1).lower()
            name = match.group(2)
            snippet: List[str] = [line]
            depth = 0
            i += 1
            while i < len(lines):
                snippet.append(lines[i])
                lowered = lines[i].lower()
                depth += lowered.count('begin')
                depth -= lowered.count('end;')
                if depth <= 0 and 'end;' in lowered:
                    break
                i += 1
            blocks.append({'name': name, 'code': '\n'.join(snippet), 'kind': kind})
        else:
            i += 1
    return blocks if blocks else [{'name': 'file', 'code': text, 'kind': 'file'}]


def extract_generic_blocks(text: str) -> List[Dict[str, str]]:
    paragraphs = [chunk.strip() for chunk in re.split(r'\n\s*\n', text) if chunk.strip()]
    return [
        {'name': f'section-{index + 1}', 'code': chunk, 'kind': 'section'}
        for index, chunk in enumerate(paragraphs)
    ] or [{'name': 'file', 'code': text, 'kind': 'file'}]


def extract_blocks(path: Path) -> List[Dict[str, str]]:
    text = read_text(path)
    suffix = path.suffix.lower()
    if suffix == '.java':
        return extract_java_blocks(text)
    if suffix == '.cob':
        return extract_cobol_blocks(text)
    if suffix == '.pas':
        return extract_delphi_blocks(text)
    return extract_generic_blocks(text)


def ensure_directories(version: str) -> None:
    dataset_dir = Path(f"src/train_weighting/dataset/{version}")
    dataset_dir.mkdir(parents=True, exist_ok=True)


def build_graph(projects_root: Path, version: str) -> Graph:
    graph = Graph()
    graph.bind("tw", TW)
    graph.bind("base", BASE_NAMESPACE)

    for language_path in projects_root.iterdir():
        if not language_path.is_dir():
            continue
        language_slug = slugify(language_path.name)
        language_uri = BASE_NAMESPACE[f"language/{language_slug}"]
        graph.add((language_uri, RDF.type, TW.Language))
        graph.add((language_uri, TW.label, Literal(language_path.name)))

        for version_path in language_path.iterdir():
            if not version_path.is_dir():
                continue
            version_slug = slugify(version_path.name)
            version_uri = BASE_NAMESPACE[f"version/{language_slug}/{version_slug}"]
            graph.add((version_uri, RDF.type, TW.ProjectVersion))
            graph.add((version_uri, TW.belongsToLanguage, language_uri))
            graph.add((version_uri, TW.label, Literal(f"{language_path.name} {version_path.name}")))

            for file_path in version_path.rglob('*'):
                if not file_path.is_file():
                    continue
                relative_path = str(file_path.relative_to(projects_root))
                file_node = file_uri(language_slug, version_slug, relative_path)
                graph.add((file_node, RDF.type, TW.File))
                graph.add((file_node, TW.path, Literal(relative_path)))
                graph.add((file_node, TW.belongsToVersion, version_uri))

                blocks = extract_blocks(file_path)
                for index, block in enumerate(blocks):
                    block_node = code_block_uri(language_slug, version_slug, relative_path, index)
                    graph.add((block_node, RDF.type, TW.CodeBlock))
                    graph.add((block_node, TW.belongsToFile, file_node))
                    graph.add((block_node, TW.belongsToVersion, version_uri))
                    graph.add((block_node, TW.belongsToLanguage, language_uri))
                    graph.add((block_node, TW.kind, Literal(block.get('kind', 'section'))))
                    graph.add((block_node, TW.name, Literal(block.get('name', 'block'))))
                    graph.add((block_node, TW.code, Literal(block['code'])))
                    graph.add((file_node, TW.containsBlock, block_node))
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RDF dataset for training weighting model")
    parser.add_argument(
        "--projects",
        default="src/train_weighting/projects",
        help="Path to the projects root directory",
    )
    parser.add_argument(
        "--version",
        default="v2",
        help="Dataset version destination (e.g. v1, v2)",
    )
    parser.add_argument(
        "--output",
        help="Destination file for the generated RDF graph",
    )
    args = parser.parse_args()

    ensure_directories(args.version)
    projects_root = Path(args.projects)
    graph = build_graph(projects_root, args.version)
    default_output = Path(f"src/train_weighting/dataset/{args.version}/library_training.ttl")
    output_path = Path(args.output) if args.output else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(output_path), format="turtle")
    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()
