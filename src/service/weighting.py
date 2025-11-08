from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Union

from rdflib import Literal, URIRef

from .Kg import KnowledgeGraph


class SimpleWeightingModel:
    def __init__(self, kg: KnowledgeGraph) -> None:
        self.kg = kg

    def score(
        self,
        query: str,
        intention: str,
        current: URIRef,
        predicate: URIRef,
        neighbor: Union[URIRef, Literal],
        path_context: Optional[List[URIRef]] = None
    ) -> float:
        label = self._candidate_text(neighbor)
        predicate_name = self.kg.local_name(predicate).lower()
        query_lower = query.lower()
        base_score = self._similarity(query_lower, label)
        base_score += self._predicate_adjustment(predicate_name, intention.lower(), query_lower)
        if path_context and neighbor in path_context:
            base_score -= 0.25
        return max(-1.0, min(1.0, base_score))

    def _candidate_text(self, node: Union[URIRef, Literal]) -> str:
        if isinstance(node, Literal):
            return str(node)
        label = self.kg.get_label(node)
        if label:
            return label
        return self.kg.local_name(node)

    def _predicate_adjustment(self, predicate_name: str, intention: str, query_lower: str) -> float:
        if 'histor' in intention and (
            predicate_name.startswith('substituida_por') or predicate_name.startswith('replaced_by')
        ):
            return -0.9
        if 'implementacaoatual' in intention and (
            predicate_name.startswith('substituida_por') or predicate_name.startswith('replaced_by')
        ):
            return 0.2
        if (
            predicate_name.startswith('tem_rota') or predicate_name.startswith('has_route')
        ) and ('rota' in query_lower or 'route' in query_lower):
            return 0.3
        return 0.0

    def _similarity(self, query: str, candidate: str) -> float:
        if not candidate:
            return -0.5
        matcher = SequenceMatcher(a=query.lower(), b=candidate.lower())
        return matcher.ratio()


class CrossEncoderWeightingModel:
    """
    Real weighting model powered by a pretrained Cross-Encoder (MS MARCO family).

    Requires the 'sentence-transformers' package which brings PyTorch as a dependency.
    Defaults to 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        device: Optional[str] = None
    ) -> None:
        self.kg = kg
        self.model_name = model_name
        self.device = device
        self._cache: Dict[Tuple[str, str, str, str, str], float] = {}
        self._backend = None  # 'sbert' or 'hf'
        # Try sentence-transformers first
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._backend = 'sbert'
            self._sbert_cls = CrossEncoder
            self._sbert_model = CrossEncoder(model_name, device=device)
            return
        except Exception:
            pass
        # Fallback to raw Transformers
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
            import torch  # type: ignore
            self._backend = 'hf'
            self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._hf_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._hf_model.to(device or 'cpu')
            self._torch = torch
        except Exception as exc:
            raise ImportError(
                'Neither sentence-transformers nor transformers+torch are available for CrossEncoderWeightingModel. '
                f'Original error: {exc}'
            )

    def _pair_texts(
        self,
        query: str,
        intention: str,
        current: URIRef,
        predicate: URIRef,
        neighbor: Union[URIRef, Literal]
    ) -> Tuple[str, str]:
        current_label = self.kg.get_label(current) or self.kg.local_name(current)
        pred_name = self.kg.local_name(predicate)
        neighbor_label = (
            str(neighbor)
            if isinstance(neighbor, Literal)
            else (self.kg.get_label(neighbor) or self.kg.local_name(neighbor))
        )
        # Phrase the pair in a way that resembles MS MARCO style relevance judging
        text_a = query
        text_b = (
            f"Current: {current_label}. Relation: {pred_name}. Target: {neighbor_label}. "
            f"Intent: {intention}."
        )
        return text_a, text_b

    def score(
        self,
        query: str,
        intention: str,
        current: URIRef,
        predicate: URIRef,
        neighbor: Union[URIRef, Literal],
        path_context: Optional[List[URIRef]] = None
    ) -> float:
        text_a, text_b = self._pair_texts(query, intention, current, predicate, neighbor)
        cache_key = (
            text_a,
            text_b,
            str(current),
            str(predicate),
            str(neighbor),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]
        # model returns higher score for more relevant pairs
        if self._backend == 'sbert':
            score = float(self._sbert_model.predict([(text_a, text_b)], convert_to_numpy=True)[0])
        else:
            # transformers forward pass
            enc = self._hf_tokenizer(text_a, text_b, return_tensors='pt', truncation=True, padding=True)
            enc = {k: v.to(self._hf_model.device) for k, v in enc.items()}
            with self._torch.no_grad():
                logits = self._hf_model(**enc).logits
                score = float(logits.squeeze().tolist())
        # optional: mild penalty for cycles
        if path_context and isinstance(neighbor, URIRef) and neighbor in path_context:
            score -= 0.25
        # Normalize to [-1,1] using tanh-like squashing for consistency with Simple model
        try:
            import math
            norm = math.tanh(score)
        except Exception:
            norm = score  # fallback, unbounded
        # cache and return
        self._cache[cache_key] = norm
        return norm


def get_default_weighting_model(kg: KnowledgeGraph):
    """
    Try to build a CrossEncoderWeightingModel. If dependencies are missing, fall back to SimpleWeightingModel.
    """
    try:
        return CrossEncoderWeightingModel(kg)
    except Exception as _:
        return SimpleWeightingModel(kg)
