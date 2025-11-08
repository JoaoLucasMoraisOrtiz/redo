from typing import Dict, List, Optional, Sequence, Union

from rdflib import Literal, URIRef

from .Kg import KnowledgeGraph
from .instruction import Instruction
from .weighting import SimpleWeightingModel


class IngestionAgent:
    def __init__(self, kg: KnowledgeGraph) -> None:
        self.kg = kg

    def store_concept(
        self,
        concept_id: str,
        label: Optional[str],
        parent_id: Optional[str],
        relation: str = 'IS_A_TYPE_OF',
        properties: Optional[Dict[str, str]] = None,
        link_direction: str = 'child_to_parent'
    ) -> URIRef:
        node = self.kg.ensure_node(concept_id, label=label)
        if parent_id:
            if link_direction == 'parent_to_child':
                self.kg.add_edge(parent_id, relation, node)
            else:
                self.kg.add_edge(node, relation, parent_id)
        if properties:
            for predicate, value in properties.items():
                self.kg.add_edge(node, predicate, value, obj_is_literal=True)
        self.kg.save()
        return node

    def update_concept(
        self,
        old_concept_id: str,
        new_concept_id: str,
        label: Optional[str] = None,
        relation: str = 'REPLACED_BY'
    ) -> URIRef:
        new_node = self.kg.ensure_node(new_concept_id, label=label)
        self.kg.add_edge(old_concept_id, relation, new_node)
        self.kg.save()
        return new_node

    def handle_instruction(self, instruction: Instruction) -> Optional[URIRef]:
        action = instruction.acao.lower()
        if action == 'guardar':
            data = instruction.metadata
            return self.store_concept(
                concept_id=data.get('concept_id', instruction.entradas[0]),
                label=data.get('label'),
                parent_id=data.get('parent_id'),
                relation=data.get('relation', 'IS_A_TYPE_OF'),
                properties=data.get('properties'),
                link_direction=data.get('link_direction', 'child_to_parent')
            )
        if action == 'atualizar' and len(instruction.entradas) >= 2:
            return self.update_concept(
                old_concept_id=instruction.entradas[0],
                new_concept_id=instruction.entradas[1],
                label=instruction.metadata.get('label'),
                relation=instruction.metadata.get('relation', 'REPLACED_BY')
            )
        return None


class RetrievalAgent:
    def __init__(
        self,
        kg: KnowledgeGraph,
        weighting_model: SimpleWeightingModel,
        max_depth: int = 4,
        min_score: float = -0.8
    ) -> None:
        self.kg = kg
        self.weighting_model = weighting_model
        self.max_depth = max_depth
        self.min_score = min_score

    def retrieve(self, instruction: Instruction) -> List[Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]]:
        entry_nodes = self._resolve_entry_nodes(instruction.entradas)
        results: List[Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]] = []
        seen: set[URIRef] = set()
        for node in entry_nodes:
            if node in seen:
                continue
            path = self._greedy_walk(instruction.raw_query, instruction.intencao, node)
            results.append({
                'entry_node': str(node),
                'entry_label': self.kg.get_label(node) or self.kg.local_name(node),
                'path': path
            })
            seen.add(node)
        return results

    def _resolve_entry_nodes(self, entradas: Sequence[str]) -> List[URIRef]:
        nodes: List[URIRef] = []
        for item in entradas:
            nodes.extend(self.kg.find_nodes(item))
        if not nodes:
            for item in entradas:
                nodes.append(self.kg.ensure_node(item, label=item))
        return nodes

    def _greedy_walk(self, query: str, intention: str, start: URIRef) -> List[Dict[str, Union[str, float]]]:
        current = start
        visited: List[URIRef] = [start]
        steps: List[Dict[str, Union[str, float]]] = []
        for _ in range(self.max_depth):
            best_step = self._next_step(query, intention, current, visited)
            if not best_step:
                break
            score, predicate, neighbor = best_step
            step_info: Dict[str, Union[str, float]] = {
                'from': str(current),
                'from_label': self.kg.get_label(current) or self.kg.local_name(current),
                'predicate': self.kg.local_name(predicate),
                'predicate_uri': str(predicate),
                'to': str(neighbor),
                'to_label': self._node_label(neighbor),
                'score': score,
            }
            steps.append(step_info)
            if isinstance(neighbor, URIRef) and neighbor not in visited:
                visited.append(neighbor)
                current = neighbor
            else:
                break
        return steps

    def _next_step(
        self,
        query: str,
        intention: str,
        current: URIRef,
        visited: List[URIRef]
    ) -> Optional[tuple[float, URIRef, Union[URIRef, Literal]]]:
        scored: List[tuple[float, URIRef, Union[URIRef, Literal]]] = []
        for predicate, neighbor in self.kg.neighbors(current):
            score = self.weighting_model.score(query, intention, current, predicate, neighbor, path_context=visited)
            scored.append((score, predicate, neighbor))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[0]
        return best if best[0] >= self.min_score else None

    def _node_label(self, node: Union[URIRef, Literal]) -> str:
        if isinstance(node, Literal):
            return str(node)
        return self.kg.get_label(node) or self.kg.local_name(node)
