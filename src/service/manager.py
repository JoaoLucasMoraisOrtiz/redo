import re
from typing import List

from .instruction import Instruction


class ManagerLLM:
    def analyze(self, query: str) -> Instruction:
        lowered = query.lower()
        action = self._detect_action(lowered)
        intention = self._detect_intention(lowered)
        entries = self._extract_entries(query)
        if not entries:
            entries = [query]
        metadata = {'detected_keywords': ','.join(entries)}
        return Instruction(acao=action, entradas=entries, intencao=intention, raw_query=query, metadata=metadata)

    def _detect_action(self, lowered_query: str) -> str:
        if any(token in lowered_query for token in ['guardar', 'salvar', 'inserir', 'store', 'save', 'ingest']):
            return 'guardar'
        if any(token in lowered_query for token in ['atualizar', 'substitu', 'sufocar', 'update', 'replace']):
            return 'atualizar'
        return 'recuperar'

    def _detect_intention(self, lowered_query: str) -> str:
        if any(keyword in lowered_query for keyword in ['histor', 'historic']):
            return 'Historica'
        if any(keyword in lowered_query for keyword in ['atual', 'corrente', 'current']):
            return 'ImplementacaoAtual'
        if any(keyword in lowered_query for keyword in ['planej', 'plan', 'planning']):
            return 'Planejamento'
        return 'Desconhecida'

    def _extract_entries(self, query: str) -> List[str]:
        quoted = re.findall(r'"([^"]+)"', query)
        if quoted:
            return quoted
        slash_terms = re.findall(r'(?:/\w[\w\-/]*)', query)
        candidates = quoted + slash_terms
        tokens = re.findall(r'(?:API\s+\w+|Funcao\w*|Function\w*|rota\s+/\w+|route\s+/\w+)', query, re.IGNORECASE)
        candidates.extend(tokens)
        cleaned = [item.strip() for item in candidates if item.strip()]
        unique = []
        for item in cleaned:
            if item not in unique:
                unique.append(item)
        return unique
