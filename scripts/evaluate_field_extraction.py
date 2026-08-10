"""
Field Extraction Evaluation Module for Receipt Parser (SROIE Dataset).

Evaluates field-level extraction performance (Exact Match %, Levenshtein Similarity %, Overall Accuracy)
across:
- RegexFieldExtractor
- LayoutHeuristicExtractor
- HybridFieldExtractor

Evaluates all 4 ground truth fields:
1. company
2. date
3. address
4. total
"""

import os
import sys
import glob
import json
import time
import re
import argparse
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    import Levenshtein
except ImportError:
    import editdistance as Levenshtein

from ocr_engine import get_ocr_engine
from field_extractor import get_field_extractor, ExtractedFields, normalize_amount

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvaluateFieldExtraction")


def compute_string_similarity(reference: str, hypothesis: str) -> float:
    """Compute normalized Levenshtein similarity score between 0.0 and 1.0."""
    ref = reference.strip().lower()
    hyp = hypothesis.strip().lower()
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    dist = Levenshtein.distance(ref, hyp)
    max_len = max(len(ref), len(hyp))
    return max(0.0, 1.0 - float(dist) / float(max_len))


def normalize_field_for_eval(field_name: str, value: str) -> str:
    """Normalize field strings for clean exact-match evaluation."""
    v = value.strip().upper()
    v = re.sub(r'\s+', ' ', v)
    if field_name == "total":
        return normalize_amount(v)
    elif field_name == "date":
        # Remove whitespace & separators for strict date comparison
        return re.sub(r'[^\d]', '', v)
    return v


def compute_exact_match(reference: str, hypothesis: str, field_name: str) -> float:
    """Check if reference and hypothesis match after normalization."""
    ref_norm = normalize_field_for_eval(field_name, reference)
    hyp_norm = normalize_field_for_eval(field_name, hypothesis)
    if not ref_norm and not hyp_norm:
        return 1.0
    if not ref_norm or not hyp_norm:
        return 0.0
    return 1.0 if ref_norm == hyp_norm else 0.0


def load_ground_truth_samples(split: str = "train", limit: int = 50) -> List[Dict[str, Any]]:
    """Load pairs of OCR inputs & ground truth field annotations."""
    annot_dir = os.path.join("data", "raw", split, "annotations")
    proc_img_dir = os.path.join("data", "processed", split, "images")
    raw_img_dir = os.path.join("data", "raw", split, "images")

    json_files = sorted(glob.glob(os.path.join(annot_dir, "receipt_*.json")))
    samples = []

    for jf in json_files:
        if jf.endswith(".ocr.json"):
            continue

        base_name = os.path.splitext(os.path.basename(jf))[0]

        with open(jf, "r", encoding="utf-8") as f:
            gt_dict = json.load(f)

        img_path = os.path.join(proc_img_dir, f"{base_name}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(proc_img_dir, f"{base_name}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(raw_img_dir, f"{base_name}.png")

        ocr_json_path = os.path.join(annot_dir, f"{base_name}.ocr.json")
        ocr_lines = []
        if os.path.exists(ocr_json_path):
            with open(ocr_json_path, "r", encoding="utf-8") as f:
                ocr_gt = json.load(f)
                # Parse bounding box words or lines
                words = ocr_gt.get("words", [])
                if words:
                    ocr_lines = [{"text": " ".join(words)}]

        samples.append({
            "id": base_name,
            "img_path": img_path,
            "gt_fields": {
                "company": gt_dict.get("company", ""),
                "date": gt_dict.get("date", ""),
                "address": gt_dict.get("address", ""),
                "total": gt_dict.get("total", "")
            },
            "ocr_lines": ocr_lines
        })

        if limit > 0 and len(samples) >= limit:
            break

    logger.info(f"Loaded {len(samples)} ground truth samples for split '{split}' (limit={limit}).")
    return samples


def evaluate_extractor(
    method_name: str,
    samples: List[Dict[str, Any]],
    ocr_engine
) -> Dict[str, Any]:
    """Benchmark a field extraction method across dataset samples."""
    extractor = get_field_extractor(method_name)

    field_keys = ["company", "date", "address", "total"]
    exact_matches = {k: [] for k in field_keys}
    similarities = {k: [] for k in field_keys}
    latencies = []

    for idx, sample in enumerate(samples):
        img_path = sample["img_path"]
        gt_fields = sample["gt_fields"]

        t0 = time.time()
        # Perform OCR extraction on receipt image
        ocr_res = ocr_engine.extract(img_path)
        ocr_data = ocr_res.to_dict()

        # Run field extraction
        pred = extractor.extract_fields(ocr_data)
        t1 = time.time()
        latencies.append(t1 - t0)

        pred_dict = pred.to_dict()

        for k in field_keys:
            gt_val = gt_fields[k]
            pred_val = pred_dict[k]

            em = compute_exact_match(gt_val, pred_val, k)
            sim = compute_string_similarity(gt_val, pred_val)

            exact_matches[k].append(em)
            similarities[k].append(sim)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            logger.info(f"Evaluated {idx + 1}/{len(samples)} samples for method '{method_name}'...")

    summary_em = {k: float(np.mean(exact_matches[k])) for k in field_keys}
    summary_sim = {k: float(np.mean(similarities[k])) for k in field_keys}

    overall_em = float(np.mean(list(summary_em.values())))
    overall_sim = float(np.mean(list(summary_sim.values())))
    avg_latency = float(np.mean(latencies))

    return {
        "method": method_name,
        "sample_count": len(samples),
        "overall_exact_match": round(overall_em, 4),
        "overall_similarity_f1": round(overall_sim, 4),
        "avg_latency_sec": round(avg_latency, 4),
        "per_field_exact_match": {k: round(v, 4) for k, v in summary_em.items()},
        "per_field_similarity_f1": {k: round(v, 4) for k, v in summary_sim.items()}
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Field Extraction algorithms on SROIE dataset.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--limit", type=int, default=50, help="Number of samples to evaluate (0 for all)")
    parser.add_argument("--ocr-engine", type=str, default="rapidocr", choices=["rapidocr", "tesseract", "easyocr"], help="OCR engine name")
    parser.add_argument("--output", type=str, default="data/field_extraction_evaluation_results.json", help="Path to save evaluation benchmark JSON")
    args = parser.parse_args()

    samples = load_ground_truth_samples(split=args.split, limit=args.limit)
    if not samples:
        logger.error("No valid ground truth samples found for evaluation!")
        sys.exit(1)

    logger.info(f"Initializing {args.ocr_engine} OCR engine for evaluation...")
    ocr_engine = get_ocr_engine(args.ocr_engine)

    methods = ["regex", "layout", "hybrid"]
    results = {}

    print("\n" + "=" * 80)
    print(f"  FIELD EXTRACTION BENCHMARK ({args.split.upper()} SPLIT - {len(samples)} SAMPLES)")
    print("=" * 80)
    print(f"{'Method':<12} | {'Overall EM':<10} | {'Overall Sim':<11} | {'Company EM':<10} | {'Date EM':<8} | {'Addr EM':<8} | {'Total EM':<8}")
    print("-" * 80)

    for method in methods:
        logger.info(f"Evaluating field extraction method '{method}'...")
        res = evaluate_extractor(method, samples, ocr_engine)
        results[method] = res

        em_dict = res["per_field_exact_match"]
        print(f"{method:<12} | {res['overall_exact_match']:<10.4f} | {res['overall_similarity_f1']:<11.4f} | {em_dict['company']:<10.4f} | {em_dict['date']:<8.4f} | {em_dict['address']:<8.4f} | {em_dict['total']:<8.4f}")

    print("=" * 80 + "\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved complete field extraction evaluation report to {args.output}")


if __name__ == "__main__":
    main()
