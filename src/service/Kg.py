import os
import re
from typing import Iterable, List, Optional, Tuple, Union

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS


class KnowledgeGraph:
    """
    Manage a Turtle-based RDF knowledge graph with helper utilities for nodes and edges.
    """

    def __init__(
        self,
        file_path: str = 'src/data/kg.rdf',
        base_namespace: str = 'http://example.org/kg/'
    ) -> None:
        self.file_path = file_path
        self.graph = Graph()
        self.base = Namespace(base_namespace)
        self.graph.bind('kg', self.base)
        if os.path.exists(self.file_path):
            try:
                self.graph.parse(self.file_path, format='turtle')
            except Exception as exc:  # Keep the in-memory graph empty if loading fails
                print(f'Failed to load graph from {self.file_path}: {exc}')

    # --- Internal utilities --------------------------------------------------
    def _slugify(self, value: str) -> str:
        cleaned = re.sub(r'[^a-zA-Z0-9]+', '_', value.strip())
        cleaned = cleaned.strip('_') or 'resource'
        return cleaned.lower()

    def _to_uri(self, value: Union[str, URIRef]) -> URIRef:
        if isinstance(value, URIRef):
            return value
        if value.startswith('http://') or value.startswith('https://'):
            return URIRef(value)
        return self.base[self._slugify(value)]

    def local_name(self, uri: Union[URIRef, str]) -> str:
        ref = uri if isinstance(uri, URIRef) else URIRef(uri)
        return ref.split('#')[-1].split('/')[-1]

    # --- Node operations ------------------------------------------------------
    def ensure_node(self, identifier: Union[str, URIRef], label: Optional[str] = None) -> URIRef:
        node_uri = self._to_uri(identifier)
        self.graph.add((node_uri, RDF.type, self.base.Concept))
        if label:
            self.graph.set((node_uri, RDFS.label, Literal(label)))
        return node_uri

    def set_label(self, identifier: Union[str, URIRef], label: str) -> None:
        node_uri = self.ensure_node(identifier)
        self.graph.set((node_uri, RDFS.label, Literal(label)))

    def get_label(self, identifier: Union[str, URIRef]) -> Optional[str]:
        node_uri = self._to_uri(identifier)
        for _, _, label in self.graph.triples((node_uri, RDFS.label, None)):
            if isinstance(label, Literal):
                return str(label)
        return None

    # --- Edge operations ------------------------------------------------------
    def add_edge(
        self,
        subject: Union[str, URIRef],
        predicate: Union[str, URIRef],
        obj: Union[str, URIRef, Literal],
        obj_is_literal: bool = False
    ) -> Tuple[URIRef, URIRef, Union[URIRef, Literal]]:
        subject_uri = self.ensure_node(subject)
        predicate_uri = self._to_uri(predicate)
        if obj_is_literal or isinstance(obj, Literal):
            object_node: Union[URIRef, Literal] = Literal(obj) if not isinstance(obj, Literal) else obj
        else:
            object_node = self.ensure_node(obj)
        self.graph.add((subject_uri, predicate_uri, object_node))
        return subject_uri, predicate_uri, object_node

    def remove_edge(
        self,
        subject: Union[str, URIRef],
        predicate: Union[str, URIRef],
        obj: Union[str, URIRef, Literal]
    ) -> None:
        subject_uri = self._to_uri(subject)
        predicate_uri = self._to_uri(predicate)
        object_node = (
            Literal(obj) if isinstance(obj, (str, Literal)) and not isinstance(obj, URIRef)
            else self._to_uri(obj)
        )
        self.graph.remove((subject_uri, predicate_uri, object_node))

    # --- Query and traversal --------------------------------------------------
    def neighbors(self, identifier: Union[str, URIRef]) -> List[Tuple[URIRef, Union[URIRef, Literal]]]:
        node_uri = self._to_uri(identifier)
        results: List[Tuple[URIRef, Union[URIRef, Literal]]] = []
        for predicate, obj in self.graph.predicate_objects(node_uri):
            if predicate in (RDF.type, RDFS.label):
                continue
            results.append((predicate, obj))
        return results

    def find_nodes(self, query: str) -> List[URIRef]:
        tokens = [token for token in query.lower().split() if len(token) > 1]
        candidates = []
        for node in set(self.graph.subjects(RDF.type, self.base.Concept)):
            label = (self.get_label(node) or '').lower()
            identifier = str(node).lower()
            score = sum(token in label or token in identifier for token in tokens)
            if score:
                candidates.append((score, node))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [node for _, node in candidates]

    def query(self, sparql_query: str):
        return self.graph.query(sparql_query)

    # --- Persistence ----------------------------------------------------------
    def save(self) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.graph.serialize(destination=self.file_path, format='turtle')

    def get_triples(self) -> List[Tuple[URIRef, URIRef, Union[URIRef, Literal]]]:
        return list(self.graph)

    def clear(self) -> None:
        self.graph = Graph()
        self.graph.bind('kg', self.base)
