from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from src.rag.vector_db import VectorDB


@dataclass(frozen=True)
class GoldLocation:
    slide: str
    page: int


def _load_gold(path: str) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".jsonl":
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    return json.loads(p.read_text(encoding="utf-8"))


def _norm_gold(row: dict) -> tuple[str, list[GoldLocation]]:
    q = row.get("query") or row.get("q")
    if not q:
        raise ValueError("Gold row missing 'query'")
    gold = row.get("gold") or row.get("locations") or row.get("answers")
    if gold is None:
        raise ValueError("Gold row missing 'gold' (list of {slide,page})")
    out: list[GoldLocation] = []
    for g in gold:
        slide = g.get("slide") or g.get("slide_file") or g.get("file")
        page = g.get("page") or g.get("page_number")
        if not slide or page is None:
            continue
        out.append(GoldLocation(slide=str(slide), page=int(page)))
    if not out:
        raise ValueError("Gold row has empty/invalid 'gold'")
    return q, out


def _is_hit(pred: dict, golds: list[GoldLocation]) -> bool:
    slide = pred.get("slide")
    page = pred.get("page")
    if not slide or page is None:
        return False
    for g in golds:
        if str(slide) == g.slide and int(page) == g.page:
            return True
    return False


def recall_at_k(rows: list[dict], *, k: int, vectordb: VectorDB) -> float:
    hits = 0
    for row in rows:
        q, golds = _norm_gold(row)
        preds = vectordb.search_locations(q, k=k, dedupe_by_page=True)
        if any(_is_hit(p, golds) for p in preds):
            hits += 1
    return hits / max(1, len(rows))


def mrr_at_k(rows: list[dict], *, k: int, vectordb: VectorDB) -> float:
    total = 0.0
    for row in rows:
        q, golds = _norm_gold(row)
        preds = vectordb.search_locations(q, k=k, dedupe_by_page=True)
        rr = 0.0
        for i, p in enumerate(preds, start=1):
            if _is_hit(p, golds):
                rr = 1.0 / i
                break
        total += rr
    return total / max(1, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="Path to gold JSON or JSONL")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    rows = _load_gold(args.gold)
    vectordb = VectorDB(documents=None)

    r = recall_at_k(rows, k=args.k, vectordb=vectordb)
    mrr = mrr_at_k(rows, k=args.k, vectordb=vectordb)

    print(json.dumps({"k": args.k, "recall@k": r, "mrr@k": mrr}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

