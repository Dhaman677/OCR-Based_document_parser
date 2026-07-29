# OCR-Based Document Parser

An end-to-end receipt & invoice processing pipeline that takes scanned receipt images, cleans and preprocesses them, performs OCR text extraction with word & line bounding boxes, and benchmarks text extraction quality against ground truth annotations.

## Current Project Status: Phase 3 Completed (OCR Extraction)

- [x] **Phase 1 — Setup & Data**: Dataset download and local bounding box exploration scripts (`SROIE` dataset).
- [x] **Phase 2 — Preprocessing**: OpenCV image processing pipeline (deskewing, bilateral denoising, CLAHE contrast, adaptive binarization).
- [x] **Phase 3 — OCR Extraction**: Unified multi-engine OCR module (RapidOCR, Tesseract, EasyOCR) and evaluation benchmark.
- [ ] **Phase 4 — Field Extraction**: Regex & NER/LLM structured field parsing (vendor, date, total, tax).
- [ ] **Phase 5 — Validation & Confidence Scoring**: Sanity checks and low-confidence flagging.
- [ ] **Phase 6 — Evaluation**: Full test set metrics computation.
- [ ] **Phase 7 — Demo App**: Interactive web UI.
- [ ] **Phase 8 — Deployment & Documentation**: Docker containerization and deployment.

---

## Benchmark Results (Phase 3 Evaluation)

Evaluated on SROIE receipt dataset split comparing **Raw Images** against **Preprocessed Images** using **RapidOCR (ONNX-based PaddleOCR architecture)**:

| Engine Configuration | Mean CER (Char Error Rate) | Mean WER (Word Error Rate) | Avg Latency (CPU) |
| :--- | :---: | :---: | :---: |
| **RapidOCR (Raw Images)** | 0.4531 | 0.7390 | 3.92s / img |
| **RapidOCR (Preprocessed Images)** | **0.4484** | **0.6370** | **3.83s / img** |

> **Key Insight**: Image preprocessing reduced the Word Error Rate (WER) from **73.90% to 63.70%** (a **10.2% absolute accuracy improvement**), demonstrating that skew correction and bilateral binarization significantly boost OCR word recognition quality on scanned thermal receipts.

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
python scripts/ocr_engine.py --image data/processed/train/images/X51005255805.png --engine rapidocr --output sample_ocr.json
```

### 3. Run Evaluation Benchmark
Evaluate OCR accuracy (CER, WER, Latency) across dataset splits:
```bash
python scripts/evaluate_ocr.py --split train --limit 50 --output data/ocr_evaluation_results.json
```

---

## Project Architecture & Directory Structure

```
OCR-extraction/
├── data/
│   ├── raw/                  # Downloaded raw images & JSON annotations
│   ├── processed/            # Preprocessed images (deskewed, binarized)
│   └── ocr_evaluation_results.json # Phase 3 evaluation benchmark results
├── scripts/
│   ├── download_data.py      # Script to fetch SROIE dataset
│   ├── explore_data.py       # Data exploration & bbox visualization
│   ├── preprocess.py         # OpenCV image preprocessing pipeline
│   ├── ocr_engine.py         # Unified OCR engine interface (RapidOCR, Tesseract, EasyOCR)
│   └── evaluate_ocr.py       # CER/WER/Latency evaluation module
├── progress.md               # Phase completion tracker
├── phases.md                 # Project roadmap
├── scope.md                  # Project scope & goals
├── sessions.md               # Work session log
├── techstack.md              # Technology stack details
└── requirements.txt          # Python dependencies
```