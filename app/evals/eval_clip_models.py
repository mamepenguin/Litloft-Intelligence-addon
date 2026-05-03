#!/usr/bin/env python3
"""CLIP model comparison eval: llm-jp-clip vs SigLIP 2.

Evaluates three CLIP-family models on image retrieval quality using
LLM-as-judge scoring (nDCG@10 / P@10 / MRR@10).

Usage:
    python eval_clip_models.py \
      --db-path /path/to/data/intelligence.db \
      --litloft-db /path/to/data/litloft.db \
      --images-dir /path/to/images \
      [--queries-file queries.yaml] \
      [--models llm-jp siglip2-256 siglip2-384] \
      [--limit 1000] \
      [--llm-api-key sk-...] \
      [--llm-base-url http://...] \
      [--llm-model gpt-4o-mini] \
      [--output results.csv]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Built-in query set (used when --queries-file is not provided)
# ---------------------------------------------------------------------------

_DEFAULT_QUERIES: list[str] = [
    # attribute + object
    "赤い車",
    "白い猫",
    "黒い服を着た人",
    "青い空と白い雲",
    "黄色い花",
    "緑の植物や木",
    "犬の写真",
    "食べ物や料理",
    # scene
    "海や川の景色",
    "本棚や書籍",
    "パソコンや作業デスク",
    "夜景や夜の街",
    "建物や建築物",
    "山や自然の風景",
    # abstract / mood
    "古い写真や古びた雰囲気",
    "子供や赤ちゃん",
    "スポーツや運動",
    "グラフや図表",
    "空白が多いシンプルな画像",
    "文字やテキストが写っている",
]

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ImageRecord:
    file_id: str
    file_path: str
    description: str  # blip_caption or vision_describe, empty string if absent


@dataclass
class SearchHit:
    rank: int  # 1-based
    file_path: str
    description: str
    score: float  # cosine similarity


@dataclass
class JudgeResult:
    label: str  # "yes" | "somewhat" | "no"
    score: float  # 1.0 / 0.5 / 0.0


@dataclass
class QueryResult:
    query: str
    hits: list[SearchHit]
    judgements: list[JudgeResult]  # parallel to hits


@dataclass
class ModelEval:
    model_key: str
    query_results: list[QueryResult]
    ndcg: float = 0.0
    precision: float = 0.0
    mrr: float = 0.0


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODEL_SPECS: dict[str, dict[str, str]] = {
    "llm-jp": {
        "kind": "open_clip",
        "name": "hf-hub:llm-jp/llm-jp-clip-vit-base-patch16",
    },
    # Standard English OpenAI CLIP (open_clip name/pretrained format)
    "clip-en-b32": {
        "kind": "open_clip",
        "name": "ViT-B-32",
        "pretrained": "openai",
    },
    "clip-en-b16": {
        "kind": "open_clip",
        "name": "ViT-B-16",
        "pretrained": "openai",
    },
    # LAION multilingual
    "laion-b32": {
        "kind": "open_clip",
        "name": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    },
    "siglip2-256": {
        "kind": "siglip2",
        "name": "google/siglip2-base-patch16-256",
    },
    "siglip2-384": {
        "kind": "siglip2",
        "name": "google/siglip2-base-patch16-384",
    },
}

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def load_image_records(
    litloft_db: Path,
    intelligence_db: Path,
    images_dir: Path,
    limit: int,
) -> list[ImageRecord]:
    """Load image file records from litloft.db, filtered to images_dir.

    Cross-references intelligence.db for blip_caption / vision_describe
    descriptions. Files missing from images_dir are silently skipped.
    """
    conn_lf = sqlite3.connect(str(litloft_db))
    conn_lf.row_factory = sqlite3.Row
    try:
        rows = conn_lf.execute(
            """
            SELECT id, file_path
            FROM files
            WHERE file_type = 'image'
              AND deleted_at IS NULL
              AND missing_since IS NULL
            ORDER BY id
            """
        ).fetchall()
    finally:
        conn_lf.close()

    images_dir_str = str(images_dir.resolve())

    # Build description map from intelligence.db (one description per file_id,
    # prefer blip_caption over vision_describe)
    desc_map: dict[str, str] = {}
    conn_int = sqlite3.connect(str(intelligence_db))
    conn_int.row_factory = sqlite3.Row
    try:
        emb_rows = conn_int.execute(
            """
            SELECT file_id, embedding_type, content_preview
            FROM embeddings
            WHERE embedding_type IN ('blip_caption', 'vision_describe')
              AND content_preview IS NOT NULL
              AND content_preview != ''
            ORDER BY
              file_id,
              CASE embedding_type
                WHEN 'vision_describe' THEN 0
                WHEN 'blip_caption'    THEN 1
              END
            """
        ).fetchall()
    finally:
        conn_int.close()

    for row in emb_rows:
        fid = row["file_id"]
        if fid not in desc_map:
            desc_map[fid] = row["content_preview"]

    records: list[ImageRecord] = []
    for row in rows:
        if len(records) >= limit:
            break
        fpath = row["file_path"]
        # Resolve to absolute path: join with images_dir when relative
        p = Path(fpath)
        full_path = p if p.is_absolute() else images_dir / fpath
        full_path = full_path.resolve()
        # Keep only files that live under images_dir and actually exist
        if not str(full_path).startswith(images_dir_str):
            continue
        if not full_path.exists():
            continue
        records.append(ImageRecord(
            file_id=row["id"],
            file_path=str(full_path),
            description=desc_map.get(row["id"], ""),
        ))

    return records


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


class OpenClipEncoder:
    """Wraps open_clip for CLIP-style models.

    Supports two loading modes:
    - hf-hub: prefix  → create_model_from_pretrained (e.g. llm-jp)
    - name + pretrained → create_model_and_transforms (e.g. ViT-B-32/openai)
    """

    def __init__(self, model_name: str, pretrained: str | None = None) -> None:
        import open_clip

        print(f"  Loading {model_name} (pretrained={pretrained}) ...", flush=True)
        if pretrained is not None:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device="cpu"
            )
            self._tokenizer = open_clip.get_tokenizer(model_name)
        else:
            self._model, self._preprocess = open_clip.create_model_from_pretrained(
                model_name, device="cpu"
            )
            self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.eval()

    def encode_images_batch(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        tensors = torch.stack([self._preprocess(img) for img in images])
        with torch.no_grad():
            feats = self._model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        import torch

        tokens = self._tokenizer([text])
        with torch.no_grad():
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze().cpu().numpy().astype(np.float32)


class SigLIP2Encoder:
    """Wraps transformers AutoModel for SigLIP 2 via joint forward pass.

    SigLIP 2's text and image encoders do not produce aligned embeddings
    when called separately (the text tokenizer omits attention_mask for
    single inputs, breaking get_text_features). The correct approach is
    the joint forward: pass both text and images together and rank by
    logits_per_image. Images are cached as PIL objects; ranking is done
    at query time in chunks of JOINT_CHUNK.
    """

    JOINT_CHUNK = 64
    # Signals to the eval loop that this encoder uses joint forward ranking
    uses_joint_forward: bool = True

    def __init__(self, model_name: str) -> None:
        from transformers import AutoModel, AutoProcessor

        print(f"  Loading {model_name} ...", flush=True)
        self._processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._pil_images: list[Image.Image] = []

    def encode_images_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Cache PIL images; return placeholder zeros (not used for ranking)."""
        self._pil_images.extend(images)
        return np.zeros((len(images), 1), dtype=np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """Not used; joint_rank is used instead."""
        return np.zeros(1, dtype=np.float32)

    def joint_rank(self, query: str, k: int = 10) -> tuple[list[int], np.ndarray]:
        """Rank all cached images by query using joint forward pass.

        Returns (top_k_indices, all_logits) where indices are into
        self._pil_images and logits are the raw SigLIP logit_per_image.
        """
        import torch

        images = self._pil_images
        all_logits: list[float] = []

        for start in range(0, len(images), self.JOINT_CHUNK):
            chunk = images[start : start + self.JOINT_CHUNK]
            inp = self._processor(
                text=[query] * len(chunk),
                images=chunk,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            with torch.no_grad():
                out = self._model(**inp)
            # logits_per_image: [n_images, n_texts]; we have 1 text → [:,0]
            all_logits.extend(out.logits_per_image[:, 0].tolist())

        arr = np.array(all_logits, dtype=np.float32)
        top_idx = np.argsort(arr)[::-1][:k].tolist()
        return top_idx, arr


def build_encoder(model_key: str) -> OpenClipEncoder | SigLIP2Encoder:
    spec = MODEL_SPECS[model_key]
    if spec["kind"] == "open_clip":
        return OpenClipEncoder(spec["name"], pretrained=spec.get("pretrained"))
    return SigLIP2Encoder(spec["name"])


# ---------------------------------------------------------------------------
# Image embedding pipeline
# ---------------------------------------------------------------------------

BATCH_SIZE = 32


def compute_image_embeddings(
    encoder: OpenClipEncoder | SigLIP2Encoder,
    records: list[ImageRecord],
) -> np.ndarray:
    """Return float32 array of shape (N, D), one row per record."""
    all_vecs: list[np.ndarray] = []

    for batch_start in range(0, len(records), BATCH_SIZE):
        batch = records[batch_start : batch_start + BATCH_SIZE]
        images: list[Image.Image] = []
        for rec in batch:
            try:
                img = Image.open(rec.file_path).convert("RGB")
                img.load()
                images.append(img)
            except Exception as e:
                print(f"  Warning: could not open {rec.file_path}: {e}", flush=True)
                # Substitute with a black 224×224 image so batch shape stays consistent
                images.append(Image.new("RGB", (224, 224)))

        vecs = encoder.encode_images_batch(images)
        all_vecs.append(vecs)

        done = min(batch_start + BATCH_SIZE, len(records))
        print(f"  {done}/{len(records)} images encoded", end="\r", flush=True)

    print(flush=True)
    return np.vstack(all_vecs).astype(np.float32)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def top_k_search(
    query_vec: np.ndarray,
    image_vecs: np.ndarray,
    records: list[ImageRecord],
    k: int = 10,
) -> list[SearchHit]:
    """Cosine similarity search. Assumes both inputs are L2-normalised."""
    scores = image_vecs @ query_vec  # shape (N,)
    top_idx = np.argsort(scores)[::-1][:k]
    hits: list[SearchHit] = []
    for rank, idx in enumerate(top_idx, start=1):
        hits.append(SearchHit(
            rank=rank,
            file_path=records[idx].file_path,
            description=records[idx].description,
            score=float(scores[idx]),
        ))
    return hits


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_TEMPLATE = (
    "クエリ「{query}」に対して、以下の画像説明は関連していますか？"
    " yes/somewhat/no で一言で答えてください。\n説明: {description}"
)

_LABEL_TO_SCORE: dict[str, float] = {
    "yes": 1.0,
    "somewhat": 0.5,
    "no": 0.0,
}


def _parse_judge_label(response: str) -> JudgeResult:
    lower = response.strip().lower()
    for label in ("somewhat", "yes", "no"):
        if label in lower:
            return JudgeResult(label=label, score=_LABEL_TO_SCORE[label])
    # Fallback: treat unrecognised responses as "no"
    return JudgeResult(label="no", score=0.0)


async def _judge_single(
    client: Any,
    llm_model: str,
    sem: asyncio.Semaphore,
    query: str,
    description: str,
) -> JudgeResult:
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        query=query,
        description=description if description else "(説明なし)",
    )
    async with sem:
        try:
            response = await client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            text = response.choices[0].message.content or ""
            return _parse_judge_label(text)
        except Exception as e:
            print(f"\n  LLM judge error: {e}", flush=True)
            return JudgeResult(label="no", score=0.0)


async def _judge_single_vision(
    client: Any,
    llm_model: str,
    sem: asyncio.Semaphore,
    query: str,
    image_path: str,
) -> JudgeResult:
    """Judge relevance by sending the actual image to a vision LLM."""
    import base64

    ext = Path(image_path).suffix.lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp", "gif": "image/gif"}.get(ext.lstrip("."), "image/jpeg")
    try:
        img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    except OSError as e:
        return JudgeResult(label="no", score=0.0)

    prompt = f"クエリ「{query}」に対してこの画像は関連していますか？ yes/somewhat/no で一言で答えてください。"
    async with sem:
        try:
            response = await client.chat.completions.create(
                model=llm_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=20,
                temperature=0.0,
            )
            text = response.choices[0].message.content or ""
            return _parse_judge_label(text)
        except Exception as e:
            print(f"\n  Vision judge error: {e}", flush=True)
            return JudgeResult(label="no", score=0.0)


async def judge_hits(
    client: Any,
    llm_model: str,
    query: str,
    hits: list[SearchHit],
    concurrency: int = 10,
    vision: bool = False,
) -> list[JudgeResult]:
    sem = asyncio.Semaphore(concurrency)
    if vision:
        tasks = [
            _judge_single_vision(client, llm_model, sem, query, hit.file_path)
            for hit in hits
        ]
    else:
        tasks = [
            _judge_single(client, llm_model, sem, query, hit.description)
            for hit in hits
        ]
    return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _dcg(scores: list[float]) -> float:
    """Discounted Cumulative Gain."""
    total = 0.0
    for i, s in enumerate(scores, start=1):
        total += s / math.log2(i + 1)
    return total


def _ideal_dcg(scores: list[float]) -> float:
    sorted_scores = sorted(scores, reverse=True)
    return _dcg(sorted_scores)


def compute_ndcg(judgements: list[JudgeResult], k: int = 10) -> float:
    scores = [j.score for j in judgements[:k]]
    dcg = _dcg(scores)
    idcg = _ideal_dcg(scores)
    if idcg < 1e-10:
        return 0.0
    return dcg / idcg


def compute_precision(judgements: list[JudgeResult], k: int = 10) -> float:
    relevant = sum(1 for j in judgements[:k] if j.score > 0)
    return relevant / min(k, len(judgements))


def compute_mrr(judgements: list[JudgeResult], k: int = 10) -> float:
    for i, j in enumerate(judgements[:k], start=1):
        if j.score > 0:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------


def load_queries(queries_file: Path | None) -> list[str]:
    if queries_file is None:
        return _DEFAULT_QUERIES

    try:
        import yaml  # type: ignore[import]
    except ImportError:
        print("Error: pyyaml is required to load --queries-file.", file=sys.stderr)
        sys.exit(1)

    data = yaml.safe_load(queries_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "queries" not in data:
        print(
            f"Error: {queries_file} must have a top-level 'queries' list.",
            file=sys.stderr,
        )
        sys.exit(1)
    return [str(q) for q in data["queries"]]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_summary(evals: list[ModelEval]) -> None:
    col_w = 16
    print("\n" + "=" * 60)
    print("CLIP model evaluation summary")
    print("=" * 60)
    header = f"{'Model':<{col_w}} {'nDCG@10':>10} {'P@10':>10} {'MRR@10':>10}"
    print(header)
    print("-" * len(header))
    for ev in evals:
        print(
            f"{ev.model_key:<{col_w}} {ev.ndcg:>10.4f} {ev.precision:>10.4f} {ev.mrr:>10.4f}"
        )
    print("=" * 60 + "\n")


def _print_top3(evals: list[ModelEval]) -> None:
    for ev in evals:
        print(f"\n--- {ev.model_key}: top-3 hits per query ---")
        for qr in ev.query_results:
            print(f"  Query: {qr.query}")
            for hit in qr.hits[:3]:
                j = qr.judgements[hit.rank - 1] if hit.rank - 1 < len(qr.judgements) else None
                label = j.label if j else "?"
                print(f"    [{hit.rank}] score={hit.score:.3f} judge={label}")
                print(f"        path: {hit.file_path}")
                if hit.description:
                    print(f"        desc: {hit.description[:120]}")


def _write_csv(evals: list[ModelEval], output: Path) -> None:
    rows: list[dict[str, str]] = []
    for ev in evals:
        for qr in ev.query_results:
            ndcg = compute_ndcg(qr.judgements)
            prec = compute_precision(qr.judgements)
            mrr = compute_mrr(qr.judgements)
            rows.append(
                {
                    "model": ev.model_key,
                    "query": qr.query,
                    "metric": "ndcg@10",
                    "value": f"{ndcg:.6f}",
                }
            )
            rows.append(
                {
                    "model": ev.model_key,
                    "query": qr.query,
                    "metric": "p@10",
                    "value": f"{prec:.6f}",
                }
            )
            rows.append(
                {
                    "model": ev.model_key,
                    "query": qr.query,
                    "metric": "mrr@10",
                    "value": f"{mrr:.6f}",
                }
            )

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "query", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results written to {output}")


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------


async def run_eval(args: argparse.Namespace) -> None:
    # --- Validate paths ---
    db_path = Path(args.db_path)
    litloft_db = Path(args.litloft_db)
    images_dir = Path(args.images_dir)

    for p, label in [
        (db_path, "--db-path"),
        (litloft_db, "--litloft-db"),
        (images_dir, "--images-dir"),
    ]:
        if not p.exists():
            print(f"Error: {label} does not exist: {p}", file=sys.stderr)
            sys.exit(1)

    queries_file = Path(args.queries_file) if args.queries_file else None
    if queries_file and not queries_file.exists():
        print(f"Error: --queries-file does not exist: {queries_file}", file=sys.stderr)
        sys.exit(1)

    queries = load_queries(queries_file)
    print(f"Loaded {len(queries)} queries")

    # --- Load image records ---
    print(f"Loading image records from {litloft_db} (limit={args.limit}) ...")
    records = load_image_records(litloft_db, db_path, images_dir, args.limit)
    if not records:
        print(
            "Error: no image records found. "
            "Check --litloft-db, --images-dir, and that files exist on disk.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Loaded {len(records)} image records")

    # --- Build OpenAI-compatible client ---
    # ollama and similar local endpoints don't require a real key;
    # AsyncOpenAI still needs a non-empty string, so fall back to "ollama".
    api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY") or (
        "ollama" if args.llm_base_url else ""
    )
    if not api_key:
        print(
            "Error: provide --llm-api-key, set OPENAI_API_KEY, or pass --llm-base-url for a local endpoint.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("Error: openai package is required (pip install openai).", file=sys.stderr)
        sys.exit(1)

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if args.llm_base_url:
        client_kwargs["base_url"] = args.llm_base_url
    llm_client = AsyncOpenAI(**client_kwargs)
    llm_model = args.llm_model

    # --- Per-model evaluation ---
    model_keys = args.models
    evals: list[ModelEval] = []

    for model_key in model_keys:
        if model_key not in MODEL_SPECS:
            print(
                f"Error: unknown model '{model_key}'. "
                f"Valid keys: {list(MODEL_SPECS)}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"\n[{model_key}] Building encoder ...")
        try:
            encoder = build_encoder(model_key)
        except Exception as e:
            print(f"Error loading model {model_key}: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"[{model_key}] Encoding {len(records)} images ...")
        image_vecs = compute_image_embeddings(encoder, records)
        joint = getattr(encoder, "uses_joint_forward", False)

        print(f"[{model_key}] Running {len(queries)} queries ...")
        query_results: list[QueryResult] = []

        for qi, query in enumerate(queries, start=1):
            print(f"  Query {qi}/{len(queries)}: {query}", flush=True)
            if joint:
                assert isinstance(encoder, SigLIP2Encoder)
                top_idx, logits = encoder.joint_rank(query, k=10)
                hits = [
                    SearchHit(
                        rank=rank + 1,
                        file_path=records[i].file_path,
                        description=records[i].description,
                        score=float(logits[i]),
                    )
                    for rank, i in enumerate(top_idx)
                ]
            else:
                query_vec = encoder.encode_text(query)
                hits = top_k_search(query_vec, image_vecs, records, k=10)

            judgements = await judge_hits(
                llm_client,
                llm_model,
                query,
                hits,
                concurrency=4 if args.vision_judge else 10,
                vision=args.vision_judge,
            )
            query_results.append(QueryResult(query=query, hits=hits, judgements=judgements))

        # Aggregate metrics
        ndcg_vals = [compute_ndcg(qr.judgements) for qr in query_results]
        prec_vals = [compute_precision(qr.judgements) for qr in query_results]
        mrr_vals = [compute_mrr(qr.judgements) for qr in query_results]

        ev = ModelEval(
            model_key=model_key,
            query_results=query_results,
            ndcg=sum(ndcg_vals) / len(ndcg_vals),
            precision=sum(prec_vals) / len(prec_vals),
            mrr=sum(mrr_vals) / len(mrr_vals),
        )
        evals.append(ev)
        print(
            f"[{model_key}] nDCG@10={ev.ndcg:.4f}  P@10={ev.precision:.4f}  MRR@10={ev.mrr:.4f}"
        )

    # --- Output ---
    _print_summary(evals)
    _print_top3(evals)

    if args.output:
        _write_csv(evals, Path(args.output))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_clip_models.py",
        description="Compare llm-jp-clip vs SigLIP 2 on image retrieval using LLM-as-judge.",
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to intelligence.db (for blip_caption / vision_describe descriptions).",
    )
    parser.add_argument(
        "--litloft-db",
        required=True,
        help="Path to litloft.db (core DB with files table).",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory prefix; only images under this path are evaluated.",
    )
    parser.add_argument(
        "--queries-file",
        default=None,
        help="YAML file with a top-level 'queries' list. Uses built-in 20 queries if omitted.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_SPECS.keys()),
        choices=list(MODEL_SPECS.keys()),
        help="Which models to evaluate (default: all three).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of images to load (default: 1000).",
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="OpenAI-compatible API key. Falls back to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Optional base URL for OpenAI-compatible API (e.g. http://localhost:11434/v1).",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="LLM model name used for judging (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write CSV results.",
    )
    parser.add_argument(
        "--vision-judge",
        action="store_true",
        default=False,
        help="Send actual images to the vision LLM for judging instead of using text descriptions.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
