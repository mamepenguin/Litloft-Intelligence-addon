#!/usr/bin/env python3
"""Image-to-image similarity eval.

Selects diverse query images from the pool, embeds all images with each model,
and finds the top-K most visually similar images for each query.
Uses a vision LLM (Ollama native, think=False) to judge visual similarity.

Usage:
    python eval_i2i.py \
      --db-path /path/to/intelligence.db \
      --litloft-db /path/to/litloft.db \
      --images-dir /path/to/images \
      --models llm-jp waon-256 clyp-v2 \
      --llm-base-url http://host.docker.internal:11434/v1 \
      --llm-model gemma4:e4b \
      [--cache-dir /path/to/cache] \
      [--n-queries 20] \
      [--topk 5]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Reuse encoder infrastructure from eval_clip_models
sys.path.insert(0, str(Path(__file__).parent))
from eval_clip_models import (
    MODEL_SPECS,
    ImageRecord,
    OllamaJudgeClient,
    _cache_path,
    build_encoder,
    compute_image_embeddings,
    load_image_records,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class I2IHit:
    rank: int
    file_path: str
    score: float
    judge_label: str  # "yes" | "somewhat" | "no"
    judge_score: float


@dataclass
class I2IQueryResult:
    query_path: str
    hits: list[I2IHit]


@dataclass
class I2IModelResult:
    model_key: str
    query_results: list[I2IQueryResult]
    ndcg: float = 0.0
    precision: float = 0.0


# ---------------------------------------------------------------------------
# Query image selection
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS = [
    ("car", ["car", "vehicle", "truck", "bus"]),
    ("animal", ["cat", "dog", "bird", "horse", "cow", "bear"]),
    ("food", ["food", "meal", "dish", "pizza", "cake", "fruit"]),
    ("sky", ["sky", "cloud", "sunset", "sunrise"]),
    ("building", ["building", "house", "church", "bridge", "architecture"]),
    ("person", ["person", "man", "woman", "child", "people"]),
    ("nature", ["forest", "tree", "mountain", "river", "beach", "grass"]),
    ("indoor", ["room", "kitchen", "office", "table", "chair", "bed"]),
    ("water", ["ocean", "sea", "lake", "river", "pool", "waterfall"]),
    ("text", ["sign", "text", "book", "screen", "poster"]),
]


def select_query_images(
    records: list[ImageRecord],
    n: int = 20,
) -> list[ImageRecord]:
    """Select diverse query images by matching BLIP captions to categories."""
    selected: list[ImageRecord] = []
    used_indices: set[int] = set()

    per_category = max(1, n // len(_CATEGORY_KEYWORDS))
    remainder = n - per_category * len(_CATEGORY_KEYWORDS)

    for _, keywords in _CATEGORY_KEYWORDS:
        count = 0
        for i, rec in enumerate(records):
            if i in used_indices:
                continue
            desc_lower = rec.description.lower()
            if any(kw in desc_lower for kw in keywords):
                selected.append(rec)
                used_indices.add(i)
                count += 1
                if count >= per_category:
                    break

    # Fill remainder with evenly-spaced records not yet selected
    step = max(1, len(records) // (n + 1))
    for i in range(0, len(records), step):
        if len(selected) >= n:
            break
        if i not in used_indices:
            selected.append(records[i])
            used_indices.add(i)

    return selected[:n]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def top_k_similar(
    query_idx: int,
    image_vecs: np.ndarray,
    records: list[ImageRecord],
    k: int = 5,
) -> list[tuple[int, float]]:
    """Return top-K indices (excluding query itself) with cosine scores."""
    q = image_vecs[query_idx]
    scores = image_vecs @ q
    scores[query_idx] = -1.0  # exclude self
    top_idx = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# LLM judge (visual similarity)
# ---------------------------------------------------------------------------


async def _judge_pair_vision(
    client: OllamaJudgeClient,
    sem: asyncio.Semaphore,
    query_path: str,
    result_path: str,
) -> tuple[str, float]:
    """Judge whether two images are visually similar."""
    try:
        q_b64 = base64.b64encode(Path(query_path).read_bytes()).decode()
        r_b64 = base64.b64encode(Path(result_path).read_bytes()).decode()
    except OSError:
        return "no", 0.0

    prompt = (
        "これら2枚の画像は視覚的に似ていますか？"
        "同じカテゴリ・主題・被写体を扱っていますか？"
        " yes/somewhat/no で一言で答えてください。"
    )
    async with sem:
        text = await client.chat_vision_pair(prompt, q_b64, r_b64)

    lower = text.strip().lower()
    for label, score in [("somewhat", 0.5), ("yes", 1.0), ("no", 0.0)]:
        if label in lower:
            return label, score
    return "no", 0.0


async def judge_i2i_hits(
    client: OllamaJudgeClient,
    query_path: str,
    hits_paths: list[str],
    concurrency: int = 4,
) -> list[tuple[str, float]]:
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _judge_pair_vision(client, sem, query_path, p)
        for p in hits_paths
    ]
    return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _dcg(scores: list[float]) -> float:
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores))


def compute_i2i_ndcg(judgements: list[tuple[str, float]], k: int = 5) -> float:
    rel = [s for _, s in judgements[:k]]
    dcg = _dcg(rel)
    idcg = _dcg(sorted(rel, reverse=True))
    return dcg / idcg if idcg > 1e-10 else 0.0


def compute_i2i_precision(judgements: list[tuple[str, float]], k: int = 5) -> float:
    return sum(1 for _, s in judgements[:k] if s > 0) / min(k, len(judgements))


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------


async def run_i2i_eval(args: argparse.Namespace) -> None:
    db_path = Path(args.db_path)
    litloft_db = Path(args.litloft_db)
    images_dir = Path(args.images_dir)

    for p, label in [(db_path, "--db-path"), (litloft_db, "--litloft-db"), (images_dir, "--images-dir")]:
        if not p.exists():
            print(f"Error: {label} does not exist: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading image records (limit={args.limit}) ...")
    records = load_image_records(litloft_db, db_path, images_dir, args.limit)
    if not records:
        print("Error: no image records found.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(records)} image records")

    query_records = select_query_images(records, n=args.n_queries)
    query_indices = {rec.file_path: i for i, rec in enumerate(records)}
    print(f"Selected {len(query_records)} query images")

    # Build Ollama client
    client = OllamaJudgeClient(args.llm_base_url, args.llm_model)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    model_results: list[I2IModelResult] = []

    for model_key in args.models:
        if model_key not in MODEL_SPECS:
            print(f"Unknown model: {model_key}", file=sys.stderr)
            sys.exit(1)

        print(f"\n[{model_key}] Building encoder ...")
        encoder = build_encoder(model_key)

        print(f"[{model_key}] Encoding {len(records)} images ...")
        image_vecs = compute_image_embeddings(
            encoder, records, cache_dir=cache_dir, model_key=model_key, images_dir=images_dir
        )

        print(f"[{model_key}] Running {len(query_records)} i2i queries ...")
        query_results: list[I2IQueryResult] = []

        for qi, qrec in enumerate(query_records, start=1):
            q_idx = query_indices[qrec.file_path]
            top_hits = top_k_similar(q_idx, image_vecs, records, k=args.topk)

            hit_paths = [records[idx].file_path for idx, _ in top_hits]
            print(f"  Query {qi}/{len(query_records)}: {Path(qrec.file_path).name}", flush=True)

            judgements = await judge_i2i_hits(client, qrec.file_path, hit_paths)

            hits = [
                I2IHit(
                    rank=rank + 1,
                    file_path=records[idx].file_path,
                    score=score,
                    judge_label=label,
                    judge_score=jscore,
                )
                for rank, ((idx, score), (label, jscore)) in enumerate(zip(top_hits, judgements))
            ]
            query_results.append(I2IQueryResult(query_path=qrec.file_path, hits=hits))

        ndcg_vals = [compute_i2i_ndcg([(h.judge_label, h.judge_score) for h in qr.hits]) for qr in query_results]
        prec_vals = [compute_i2i_precision([(h.judge_label, h.judge_score) for h in qr.hits]) for qr in query_results]

        result = I2IModelResult(
            model_key=model_key,
            query_results=query_results,
            ndcg=sum(ndcg_vals) / len(ndcg_vals),
            precision=sum(prec_vals) / len(prec_vals),
        )
        model_results.append(result)
        print(f"[{model_key}] i2i nDCG@{args.topk}={result.ndcg:.4f}  P@{args.topk}={result.precision:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Image-to-image similarity eval (top-{args.topk})")
    print("=" * 60)
    col = 16
    print(f"{'Model':<{col}} {'nDCG':>10} {'P@K':>10}")
    print("-" * (col + 22))
    for r in model_results:
        print(f"{r.model_key:<{col}} {r.ndcg:>10.4f} {r.precision:>10.4f}")
    print("=" * 60)

    # Top-3 detail
    for r in model_results:
        print(f"\n--- {r.model_key}: top-3 per query ---")
        for qr in r.query_results:
            print(f"  Query: {Path(qr.query_path).name}")
            for hit in qr.hits[:3]:
                print(f"    [{hit.rank}] score={hit.score:.3f} judge={hit.judge_label} {Path(hit.file_path).name}")

    await client.aclose()


# ---------------------------------------------------------------------------
# OllamaJudgeClient extension for pair vision
# ---------------------------------------------------------------------------

# Monkey-patch pair vision onto OllamaJudgeClient
async def _chat_vision_pair(self: OllamaJudgeClient, prompt: str, img1_b64: str, img2_b64: str) -> str:
    import httpx
    body = {
        "model": self._model,
        "messages": [{"role": "user", "content": prompt, "images": [img1_b64, img2_b64]}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 20},
    }
    try:
        resp = await self._http.post(f"{self._base_url}/api/chat", json=body)
        return resp.json().get("message", {}).get("content", "")
    except Exception:
        return ""


OllamaJudgeClient.chat_vision_pair = _chat_vision_pair  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval_i2i.py",
        description="Image-to-image similarity eval using visual LLM judging.",
    )
    p.add_argument("--db-path", required=True, help="Path to intelligence.db")
    p.add_argument("--litloft-db", required=True, help="Path to litloft.db (files table)")
    p.add_argument("--images-dir", required=True, help="Images directory prefix")
    p.add_argument("--models", nargs="+", default=["llm-jp", "waon-256", "clyp-v2"],
                   choices=list(MODEL_SPECS.keys()))
    p.add_argument("--limit", type=int, default=1000, help="Max images in pool")
    p.add_argument("--n-queries", type=int, default=20, help="Number of query images")
    p.add_argument("--topk", type=int, default=5, help="Top-K similar images per query")
    p.add_argument("--llm-base-url", required=True)
    p.add_argument("--llm-model", default="gemma4:e4b")
    p.add_argument("--cache-dir", default=None, help="Embedding cache directory")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    asyncio.run(run_i2i_eval(args))


if __name__ == "__main__":
    main()
