#!/usr/bin/env python3
"""Memory and speed benchmark for CLIP-family models.

Measures:
  - Model load time (seconds)
  - RSS after load (MB)
  - Image encode throughput (images/sec) for 1000 images
  - Peak RSS during image encode (MB)
  - Text encode latency (ms/query) for 20 queries

Usage:
    python benchmark_clip.py \
      --litloft-db /path/to/litloft.db \
      --images-dir /path/to/images \
      --models llm-jp waon-256 clyp-v2 \
      [--limit 1000] \
      [--text-queries 20]
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from eval_clip_models import MODEL_SPECS, build_encoder, load_image_records

_TEXT_QUERIES_JA = [
    "赤い車", "白い猫", "黒い服を着た人", "青い空と白い雲", "黄色い花",
    "緑の植物や木", "犬の写真", "食べ物や料理", "海や川の景色", "本棚や書籍",
    "パソコンや作業デスク", "夜景や夜の街", "建物や建築物", "山や自然の風景",
    "古い写真や古びた雰囲気", "子供や赤ちゃん", "スポーツや運動", "グラフや図表",
    "空白が多いシンプルな画像", "文字やテキストが写っている",
]


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def _rss_mb() -> float:
    """Current RSS in MB via /proc/self/status."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    # macOS fallback via resource
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS returns bytes, Linux returns KB
        import platform
        if platform.system() == "Darwin":
            return rss / (1024 * 1024)
        return rss / 1024
    except Exception:
        return 0.0


def _gc_collect() -> None:
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

BATCH_SIZE = 32


def _load_images(records: list, limit: int) -> list[Image.Image]:
    images = []
    for rec in records[:limit]:
        try:
            img = Image.open(rec.file_path).convert("RGB")
            img.load()
            images.append(img)
        except Exception:
            images.append(Image.new("RGB", (224, 224)))
    return images


def benchmark_model(
    model_key: str,
    images: list[Image.Image],
    text_queries: list[str],
) -> dict:
    _gc_collect()
    rss_before = _rss_mb()

    # --- Load ---
    t0 = time.perf_counter()
    encoder = build_encoder(model_key)
    load_time = time.perf_counter() - t0
    rss_after_load = _rss_mb()

    # --- Image encode ---
    n = len(images)
    rss_peak = rss_after_load
    t0 = time.perf_counter()
    all_vecs = []
    for start in range(0, n, BATCH_SIZE):
        batch = images[start: start + BATCH_SIZE]
        vecs = encoder.encode_images_batch(batch)
        all_vecs.append(vecs)
        rss_peak = max(rss_peak, _rss_mb())
    img_encode_time = time.perf_counter() - t0
    img_throughput = n / img_encode_time if img_encode_time > 0 else 0.0

    # --- Text encode ---
    latencies = []
    for q in text_queries:
        t0 = time.perf_counter()
        encoder.encode_text(q)
        latencies.append((time.perf_counter() - t0) * 1000)
    avg_text_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "model": model_key,
        "load_time_s": round(load_time, 2),
        "rss_after_load_mb": round(rss_after_load, 1),
        "rss_delta_load_mb": round(rss_after_load - rss_before, 1),
        "img_encode_time_s": round(img_encode_time, 2),
        "img_throughput_per_s": round(img_throughput, 1),
        "rss_peak_encode_mb": round(rss_peak, 1),
        "rss_delta_encode_mb": round(rss_peak - rss_after_load, 1),
        "text_latency_ms_avg": round(avg_text_latency, 2),
        "text_latency_ms_min": round(min(latencies), 2) if latencies else 0,
        "text_latency_ms_max": round(max(latencies), 2) if latencies else 0,
        "n_images": n,
        "n_text_queries": len(text_queries),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmark_clip.py",
        description="Memory and speed benchmark for CLIP-family models.",
    )
    p.add_argument("--litloft-db", required=True, help="Path to litloft.db")
    p.add_argument("--images-dir", required=True, help="Images directory prefix")
    p.add_argument(
        "--models", nargs="+",
        default=["llm-jp", "waon-256", "clyp-v2"],
        choices=list(MODEL_SPECS.keys()),
    )
    p.add_argument("--limit", type=int, default=1000, help="Images to encode")
    p.add_argument("--text-queries", type=int, default=20, help="Text queries for latency test")
    # dummy db path (load_image_records requires it but benchmark doesn't use descriptions)
    p.add_argument("--db-path", default="/dev/null", help="intelligence.db (descriptions, optional)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    images_dir = Path(args.images_dir)
    litloft_db = Path(args.litloft_db)
    db_path = Path(args.db_path) if args.db_path != "/dev/null" else litloft_db

    print(f"Loading image records (limit={args.limit}) ...")
    # Use litloft_db as intelligence_db fallback for description query
    records = load_image_records(litloft_db, db_path, images_dir, args.limit)
    if not records:
        print("Error: no image records found.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(records)} records. Pre-loading images ...")
    images = _load_images(records, args.limit)
    print(f"Pre-loaded {len(images)} images into memory")

    text_queries = _TEXT_QUERIES_JA[: args.text_queries]
    results = []

    for model_key in args.models:
        print(f"\n{'='*50}")
        print(f"Benchmarking: {model_key}")
        print(f"{'='*50}")
        try:
            r = benchmark_model(model_key, images, text_queries)
            results.append(r)
            print(f"  Load time:        {r['load_time_s']:.2f}s")
            print(f"  RSS after load:   {r['rss_after_load_mb']:.1f} MB  (+{r['rss_delta_load_mb']:.1f} MB)")
            print(f"  Image throughput: {r['img_throughput_per_s']:.1f} imgs/s  ({r['img_encode_time_s']:.2f}s for {r['n_images']} imgs)")
            print(f"  Peak RSS encode:  {r['rss_peak_encode_mb']:.1f} MB  (+{r['rss_delta_encode_mb']:.1f} MB)")
            print(f"  Text latency:     {r['text_latency_ms_avg']:.2f} ms/query  (min={r['text_latency_ms_min']:.2f}, max={r['text_latency_ms_max']:.2f})")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            results.append({"model": model_key, "error": str(e)})
        _gc_collect()

    # Summary table
    print("\n" + "=" * 90)
    print("Benchmark summary")
    print("=" * 90)
    hdr = f"{'Model':<16} {'Load(s)':>8} {'RSS_load(MB)':>13} {'Imgs/s':>8} {'PeakRSS(MB)':>13} {'TextLatency(ms)':>16}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"{r['model']:<16}  ERROR: {r['error']}")
        else:
            print(
                f"{r['model']:<16}"
                f" {r['load_time_s']:>8.2f}"
                f" {r['rss_after_load_mb']:>13.1f}"
                f" {r['img_throughput_per_s']:>8.1f}"
                f" {r['rss_peak_encode_mb']:>13.1f}"
                f" {r['text_latency_ms_avg']:>16.2f}"
            )
    print("=" * 90)


if __name__ == "__main__":
    main()
