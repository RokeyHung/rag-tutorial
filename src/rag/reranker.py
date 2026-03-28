from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RerankResult:
    index: int
    score: float


class CrossEncoderReranker:
    """
    Re-ranker dùng Cross-Encoder (Retrieve-many -> Re-rank-few).
    """

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, texts: list[str]) -> list[RerankResult]:
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs)
        out: list[RerankResult] = []
        for i, s in enumerate(scores):
            try:
                out.append(RerankResult(index=i, score=float(s)))
            except Exception:
                out.append(RerankResult(index=i, score=s))
        out.sort(key=lambda x: x.score, reverse=True)
        return out

