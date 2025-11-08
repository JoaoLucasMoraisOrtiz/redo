from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Instruction:
    acao: str
    entradas: List[str]
    intencao: str
    raw_query: str
    metadata: Dict[str, str] = field(default_factory=dict)
