"""
OCR Evaluation Module for Receipt Parser.

Calculates Character Error Rate (CER), Word Error Rate (WER), exact match accuracy,
and average extraction latency across:
- Raw Images vs Preprocessed Images
- Tesseract vs RapidOCR / EasyOCR engines
"""

import os
import sys
import json
import glob
import time
import argparse
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    import Levenshtein
except ImportError:
    import editdistance as Levenshtein  # fallback if needed

from ocr_engine import get_ocr_engine, find_tesseract_cmd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvaluateOCR")


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER)."""
    ref = reference.strip()
    hyp = hypothesis.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    dist = Levenshtein.distance(ref, hyp)
    return float(dist) / float(len(ref))


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER)."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = Levenshtein.distance(ref_words, hyp_words)
    return float(dist) / float(len(ref_words))


def load_dataset_samples(split: str = "train", limit: int = 50) -> List[Dict[str, Any]]:
    """Load pairs of (raw_img_path, proc_img_path, ground_truth_ocr_json)."""
    raw_img_dir = os.path.join("data", "raw", split, "images")
    proc_img_dir = os.path.join("data", "processed", split, "images")
    annot_dir = os.path.join("data", "raw", split, "annotations")

    raw_images = sorted(glob.glob(os.path.join(raw_img_dir, "*.png")) + glob.glob(os.path.join(raw_img_dir, "*.jpg")))
    samples = []

    for raw_path in raw_images:
        base_name = os.path.splitext(os.path.basename(raw_path))[0]
        proc_path = os.path.join(proc_img_dir, f"{base_name}.png")
        if not os.path.exists(proc_path):
            proc_path = os.path.join(proc_img_dir, f"{base_name}.jpg")

        ocr_json_path = os.path.join(annot_dir, f"{base_name}.ocr.json")
        if not os.path.exists(ocr_json_path):
            continue

        with open(ocr_json_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        gt_words = gt_data.get("words", [])
        gt_text = " ".join(gt_words)

        samples.append({
            "id": base_name,
            "raw_img": raw_path,
            "proc_img": proc_path if os.path.exists(proc_path) else raw_path,
            "gt_text": gt_text,
            "gt_words": gt_words
        })

        if limit > 0 and len(samples) >= limit:
            break

    logger.info(f"Loaded {len(samples)} samples for split '{split}' (limit={limit}).")
    return samples


def evaluate_engine_config(
    engine_name: str,
    samples: List[Dict[str, Any]],
    use_preprocessed: bool = True
) -> Dict[str, Any]:
    """Run OCR extraction and evaluate CER, WER, and execution speed."""
    engine = get_ocr_engine(engine_name)
    img_key = "proc_img" if use_preprocessed else "raw_img"

    cer_list = []
    wer_list = []
    times = []
    sample_results = []

    for idx, sample in enumerate(samples):
        img_path = sample[img_key]
        gt_text = sample["gt_text"]

        t0 = time.time()
        result = engine.extract(img_path)
        t1 = time.time()

        pred_text = result.full_text
        cer = compute_cer(gt_text, pred_text)
        wer = compute_wer(gt_text, pred_text)

        cer_list.append(cer)
        wer_list.append(wer)
        times.append(t1 - t0)

        sample_results.append({
            "id": sample["id"],
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "time_sec": round(t1 - t0, 3)
        })
        if (idx + 1) % 5 == 0 or (idx + 1) == len(samples):
            logger.info(f"Processed {idx + 1}/{len(samples)} samples for {engine_name} (preprocessed={use_preprocessed})...")

    avg_cer = float(np.mean(cer_list)) if cer_list else 0.0
    avg_wer = float(np.mean(wer_list)) if wer_list else 0.0
    avg_time = float(np.mean(times)) if times else 0.0

    return {
        "engine": engine_name,
        "preprocessed": use_preprocessed,
        "sample_count": len(samples),
        "mean_cer": round(avg_cer, 4),
        "mean_wer": round(avg_wer, 4),
        "avg_latency_sec": round(avg_time, 4),
        "sample_results": sample_results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR engines on receipt dataset.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to evaluate")
    parser.add_argument("--limit", type=int, default=50, help="Number of samples to evaluate (0 for all)")
    parser.add_argument("--output", type=str, default="data/ocr_evaluation_results.json", help="Path to save evaluation JSON")
    args = parser.parse_args()

    samples = load_dataset_samples(split=args.split, limit=args.limit)
    if not samples:
        logger.error("No valid dataset samples found for evaluation!")
        sys.exit(1)

    eval_configs = []

    # Check Tesseract availability
    try:
        if find_tesseract_cmd():
            get_ocr_engine("tesseract")
            eval_configs.append(("tesseract", False, "Tesseract (Raw Images)"))
            eval_configs.append(("tesseract", True, "Tesseract (Preprocessed Images)"))
        else:
            logger.info("Tesseract binary not found in system PATH; skipping Tesseract benchmark.")
    except Exception as e:
        logger.info(f"Tesseract engine not available for benchmark: {e}")

    # Try adding RapidOCR if available
    try:
        get_ocr_engine("rapidocr")
        eval_configs.append(("rapidocr", False, "RapidOCR (Raw Images)"))
        eval_configs.append(("rapidocr", True, "RapidOCR (Preprocessed Images)"))
    except Exception as e:
        logger.info(f"RapidOCR engine not available for benchmark: {e}")

    # Try adding EasyOCR if available
    try:
        get_ocr_engine("easyocr")
        eval_configs.append(("easyocr", False, "EasyOCR (Raw Images)"))
        eval_configs.append(("easyocr", True, "EasyOCR (Preprocessed Images)"))
    except Exception as e:
        logger.info(f"EasyOCR engine not available for benchmark: {e}")

    results = {}
    print("\n" + "=" * 70)
    print(f"  OCR ENGINE EVALUATION BENCHMARK ({args.split.upper()} SPLIT - {len(samples)} SAMPLES)")
    print("=" * 70)
    print(f"{'Engine & Configuration':<35} | {'Mean CER':<10} | {'Mean WER':<10} | {'Latency (s)':<10}")
    print("-" * 70)

    for eng_name, is_proc, label in eval_configs:
        try:
            logger.info(f"Evaluating {label}...")
            res = evaluate_engine_config(eng_name, samples, use_preprocessed=is_proc)
            results[label] = res
            print(f"{label:<35} | {res['mean_cer']:<10.4f} | {res['mean_wer']:<10.4f} | {res['avg_latency_sec']:<10.4f}")
        except Exception as e:
            logger.error(f"Failed to evaluate {label}: {e}")

    print("=" * 70 + "\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved complete evaluation report to {args.output}")


if __name__ == "__main__":
    main()
