# OCR-Based Document Parser

An end-to-end receipt & invoice processing pipeline that takes scanned receipt images, cleans and preprocesses them, performs OCR text extraction with word & line bounding boxes, and benchmarks text extraction quality against ground truth annotations.

## Current Project Status: Phase 5 Completed (Validation & Confidence Scoring)

- [x] **Phase 1 — Setup & Data**: Dataset download and local bounding box exploration scripts (`SROIE` dataset).
- [x] **Phase 2 — Preprocessing**: OpenCV image processing pipeline (deskewing, bilateral denoising, CLAHE contrast, adaptive binarization).
- [x] **Phase 3 — OCR Extraction**: Unified multi-engine OCR module (RapidOCR, Tesseract, EasyOCR) and evaluation benchmark.
- [x] **Phase 4 — Field Extraction**: Regex, Layout Heuristic, and Hybrid ensemble field extraction (`company`, `date`, `address`, `total`).
- [x] **Phase 5 — Validation & Confidence Scoring**: Rule-based sanity checks, per-field dynamic confidence scoring, ISO date formatting, and low-confidence review flagging.
- [ ] **Phase 6 — Evaluation**: Full test set metrics computation.
- [ ] **Phase 7 — Demo App**: Interactive web UI.
- [ ] **Phase 8 — Deployment & Documentation**: Docker containerization and deployment.

---

## Benchmark Results

### 1. OCR Extraction Benchmark (Phase 3)
Evaluated on SROIE receipt dataset split comparing **Raw Images** against **Preprocessed Images** using **RapidOCR (ONNX-based PaddleOCR architecture)**:

| Engine Configuration | Mean CER (Char Error Rate) | Mean WER (Word Error Rate) | Avg Latency (CPU) |
| :--- | :---: | :---: | :---: |
| **RapidOCR (Raw Images)** | 0.4531 | 0.7390 | 3.92s / img |
| **RapidOCR (Preprocessed Images)** | **0.4484** | **0.6370** | **3.83s / img** |

> **Key Insight**: Image preprocessing reduced the Word Error Rate (WER) from **73.90% to 63.70%** (a **10.2% absolute accuracy improvement**), demonstrating that skew correction and bilateral binarization significantly boost OCR word recognition quality on scanned thermal receipts.

### 2. Field Extraction Benchmark (Phase 4)
Evaluated on 50 SROIE receipt samples comparing **Regex**, **Layout Heuristic**, and **Hybrid Ensemble** field extractors across 4 target fields (`company`, `date`, `address`, `total`):

| Extraction Method | Overall Exact Match | Overall Similarity (F1) | Company Sim | Date Sim | Address Sim | Total Sim |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regex Extractor** | 34.00% | 71.20% | 79.66% | 67.78% | 70.09% | 67.27% |
| **Layout Heuristic Extractor** | 27.50% | 68.39% | 79.07% | 69.40% | 70.73% | 54.37% |
| **Hybrid Ensemble Extractor** | **34.00%** | **71.49%** | **80.82%** | **67.78%** | **70.09%** | **67.27%** |

> **Key Insight**: The **Hybrid Ensemble Extractor** achieved the highest overall string similarity F1 score of **71.49%**, leveraging pattern constraints for dates/totals and spatial top-line heuristics for company names (80.82% similarity).

### 3. Validation & Confidence Scoring Benchmark (Phase 5)
Evaluated across 50 SROIE receipt samples using `ReceiptValidator` with a confidence threshold of **0.70**:

| Metric Name | Score / Value | Description |
| :--- | :---: | :--- |
| **Mean Overall Confidence Score** | **0.7828** | Weighted receipt extraction confidence |
| **Valid Receipts Pass Rate** | **44.0%** | Clean receipts passing all validation checks |
| **Requires Human Review Flag Rate** | **56.0%** | Receipts flagged for low confidence or rule warnings |

#### Dynamic Confidence vs. Extraction Accuracy Correlation:

| Field Name | High-Confidence Count | High-Confidence F1 Sim | Low-Confidence Count | Low-Confidence F1 Sim |
| :--- | :---: | :---: | :---: | :---: |
| **Date** | 29 | **95.86%** | 21 | **29.00%** |
| **Address** | 45 | **77.44%** | 5 | **3.94%** |
| **Total** | 45 | **69.19%** | 5 | **50.00%** |
| **Company** | 50 | **80.82%** | 0 | N/A |

> **Key Insight**: Dynamic per-field confidence scoring strongly correlates with actual extraction accuracy. For example, dates with confidence $\ge 0.70$ achieved **95.86% similarity F1**, whereas low-confidence dates dropped to **29.00% F1**, allowing the pipeline to reliably flag uncertain extractions for human review.

---

## Installation & Setup

1. **Clone the repository and set up environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Download Dataset**:
   ```bash
   python scripts/download_data.py
   ```

---

## Usage Guide

### 1. Preprocess Receipt Images
Run the OpenCV preprocessing pipeline on dataset images:
```bash
python scripts/preprocess.py --split train
```

### 2. Extract OCR from a Single Image
Run OCR extraction using RapidOCR (or Tesseract / EasyOCR):
```bash
python scripts/ocr_engine.py --image data/raw/train/images/receipt_train_0000.jpg --engine rapidocr --output sample_ocr.json
```

### 3. Run Field Extraction & Validation on a Receipt
Extract structured fields (`company`, `date`, `address`, `total`) and perform rule-based validation & confidence scoring:
```bash
python scripts/field_extractor.py --image data/raw/train/images/receipt_train_0000.jpg --method hybrid --threshold 0.70 --output fields_output.json
```

### 4. Run Evaluation Benchmarks
- **OCR Engine Benchmark**:
  ```bash
  python scripts/evaluate_ocr.py --split train --limit 50 --output data/ocr_evaluation_results.json
  ```

- **Field Extraction Benchmark**:
  ```bash
  python scripts/evaluate_field_extraction.py --split train --limit 50 --output data/field_extraction_evaluation_results.json
  ```

- **Validation & Confidence Scoring Benchmark**:
  ```bash
  python scripts/evaluate_validation.py --split train --limit 50 --threshold 0.70 --output data/validation_evaluation_results.json
  ```

---

## Project Architecture & Directory Structure

```
OCR-extraction/
├── data/
│   ├── raw/                            # Downloaded raw images & JSON annotations
│   ├── processed/                      # Preprocessed images (deskewed, binarized)
│   ├── ocr_evaluation_results.json     # Phase 3 OCR evaluation benchmark results
│   ├── field_extraction_evaluation_results.json # Phase 4 field extraction benchmarks
│   └── validation_evaluation_results.json       # Phase 5 validation & confidence benchmarks
├── scripts/
│   ├── download_data.py                # Script to fetch SROIE dataset
│   ├── explore_data.py                 # Data exploration & bbox visualization
│   ├── preprocess.py                   # OpenCV image preprocessing pipeline
│   ├── ocr_engine.py                   # Unified OCR engine interface (RapidOCR, Tesseract, EasyOCR)
│   ├── evaluate_ocr.py                 # CER/WER/Latency OCR evaluation module
│   ├── field_extractor.py              # Structured field extractor (Regex, Layout, Hybrid)
│   ├── evaluate_field_extraction.py    # Exact Match & Levenshtein Similarity benchmark script
│   ├── validator.py                    # Rule-based validator & dynamic confidence scorer
│   └── evaluate_validation.py          # Validation & review flagging evaluation module
├── progress.md                         # Phase completion tracker
├── phases.md                           # Project roadmap
├── scope.md                            # Project scope & goals
├── sessions.md                         # Work session log
├── techstack.md                        # Technology stack details
└── requirements.txt                    # Python dependencies
```