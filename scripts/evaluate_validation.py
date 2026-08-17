"""
Validation & Confidence Scoring Evaluation Module for Receipt Parser (SROIE Dataset).

Benchmarks rule-based validation checks, confidence scoring accuracy,
review flag distribution, and high- vs low-confidence field accuracy across SROIE samples.
"""

import os
import sys
import glob
import json
import time
import argparse
import logging
from typing import List, Dict, Any
import numpy as np

from ocr_engine import get_ocr_engine
from field_extractor import get_field_extractor
from validator import ReceiptValidator
from evaluate_field_extraction import load_ground_truth_samples, compute_exact_match, compute_string_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvaluateValidation")


def evaluate_validation_performance(
    samples: List[Dict[str, Any]],
    ocr_engine,
    confidence_threshold: float = 0.70
) -> Dict[str, Any]:
    """Run validation benchmark on receipt ground-truth dataset."""
    extractor = get_field_extractor("hybrid")
    validator = ReceiptValidator(confidence_threshold=confidence_threshold)

    field_keys = ["company", "date", "address", "total"]

    overall_confidences = []
    per_field_confidences = {k: [] for k in field_keys}

    high_conf_exact_matches = {k: [] for k in field_keys}
    high_conf_similarities = {k: [] for k in field_keys}
    low_conf_exact_matches = {k: [] for k in field_keys}
    low_conf_similarities = {k: [] for k in field_keys}

    requires_review_count = 0
    valid_count = 0
    all_flag_counts = {}

    for idx, sample in enumerate(samples):
        img_path = sample["img_path"]
        gt_fields = sample["gt_fields"]

        # Run OCR and Field Extraction
        ocr_res = ocr_engine.extract(img_path)
        ocr_data = ocr_res.to_dict()
        pred_fields = extractor.extract_fields(ocr_data)

        # Run Validator
        val_res = validator.validate_fields(
            {"company": pred_fields.company, "date": pred_fields.date, "address": pred_fields.address, "total": pred_fields.total},
            raw_text=ocr_data.get("full_text", ""),
            method_confidences=pred_fields.confidences
        )

        overall_confidences.append(val_res.overall_confidence)
        if val_res.requires_review:
            requires_review_count += 1
        if val_res.is_valid:
            valid_count += 1

        for flag in val_res.flags:
            all_flag_counts[flag] = all_flag_counts.get(flag, 0) + 1

        for k in field_keys:
            gt_val = gt_fields[k]
            pred_val = getattr(pred_fields, k)
            conf_val = val_res.confidences[k]

            em = compute_exact_match(gt_val, pred_val, k)
            sim = compute_string_similarity(gt_val, pred_val)

            per_field_confidences[k].append(conf_val)

            if conf_val >= confidence_threshold:
                high_conf_exact_matches[k].append(em)
                high_conf_similarities[k].append(sim)
            else:
                low_conf_exact_matches[k].append(em)
                low_conf_similarities[k].append(sim)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            logger.info(f"Evaluated {idx + 1}/{len(samples)} samples for validation benchmark...")

    avg_overall_conf = float(np.mean(overall_confidences))
    flag_rate = float(requires_review_count) / float(len(samples))
    valid_rate = float(valid_count) / float(len(samples))

    avg_field_conf = {k: float(np.mean(per_field_confidences[k])) for k in field_keys}

    high_conf_summary = {}
    low_conf_summary = {}

    for k in field_keys:
        high_conf_summary[k] = {
            "sample_count": len(high_conf_similarities[k]),
            "exact_match": round(float(np.mean(high_conf_exact_matches[k])), 4) if high_conf_exact_matches[k] else 0.0,
            "similarity_f1": round(float(np.mean(high_conf_similarities[k])), 4) if high_conf_similarities[k] else 0.0
        }
        low_conf_summary[k] = {
            "sample_count": len(low_conf_similarities[k]),
            "exact_match": round(float(np.mean(low_conf_exact_matches[k])), 4) if low_conf_exact_matches[k] else 0.0,
            "similarity_f1": round(float(np.mean(low_conf_similarities[k])), 4) if low_conf_similarities[k] else 0.0
        }

    return {
        "sample_count": len(samples),
        "confidence_threshold": confidence_threshold,
        "mean_overall_confidence": round(avg_overall_conf, 4),
        "review_flag_rate": round(flag_rate, 4),
        "valid_receipt_rate": round(valid_rate, 4),
        "per_field_mean_confidence": {k: round(v, 4) for k, v in avg_field_conf.items()},
        "high_confidence_accuracy": high_conf_summary,
        "low_confidence_accuracy": low_conf_summary,
        "top_warning_flags": dict(sorted(all_flag_counts.items(), key=lambda x: x[1], reverse=True))
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Validation and Confidence Scoring module on SROIE dataset.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--limit", type=int, default=50, help="Number of samples to evaluate (0 for all)")
    parser.add_argument("--ocr-engine", type=str, default="rapidocr", choices=["rapidocr", "tesseract", "easyocr"], help="OCR engine name")
    parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold for flagging")
    parser.add_argument("--output", type=str, default="data/validation_evaluation_results.json", help="Path to save output benchmark JSON")
    args = parser.parse_args()

    samples = load_ground_truth_samples(split=args.split, limit=args.limit)
    if not samples:
        logger.error("No valid ground truth samples found for evaluation!")
        sys.exit(1)

    logger.info(f"Initializing {args.ocr_engine} OCR engine...")
    ocr_engine = get_ocr_engine(args.ocr_engine)

    logger.info("Running validation evaluation benchmark...")
    res = evaluate_validation_performance(samples, ocr_engine, confidence_threshold=args.threshold)

    print("\n" + "=" * 70)
    print(f"  VALIDATION & CONFIDENCE SCORING BENCHMARK ({args.split.upper()} SPLIT)")
    print("=" * 70)
    print(f"Total Receipt Samples Evaluated:  {res['sample_count']}")
    print(f"Confidence Threshold:            {res['confidence_threshold']:.2f}")
    print(f"Mean Overall Confidence Score:   {res['mean_overall_confidence']:.4f}")
    print(f"Valid Receipts Pass Rate:        {res['valid_receipt_rate'] * 100:.1f}%")
    print(f"Requires Human Review Flag Rate: {res['review_flag_rate'] * 100:.1f}%")
    print("-" * 70)
    print("Per-field Mean Confidences:")
    for k, v in res["per_field_mean_confidence"].items():
        print(f"  - {k:<10}: {v:.4f}")
    print("-" * 70)
    print(f"{'Field':<10} | {'High-Conf Count':<15} | {'High-Conf Sim F1':<16} | {'Low-Conf Count':<15} | {'Low-Conf Sim F1':<15}")
    print("-" * 70)
    for k in ["company", "date", "address", "total"]:
        hc = res["high_confidence_accuracy"][k]
        lc = res["low_confidence_accuracy"][k]
        print(f"{k:<10} | {hc['sample_count']:<15} | {hc['similarity_f1']:<16.4f} | {lc['sample_count']:<15} | {lc['similarity_f1']:<15.4f}")
    print("=" * 70)
    print("Top Warning Flags Triggered:")
    for flag, count in list(res["top_warning_flags"].items())[:8]:
        print(f"  - {flag:<35}: {count}")
    print("=" * 70 + "\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    logger.info(f"Saved complete validation benchmark report to {args.output}")


if __name__ == "__main__":
    main()
