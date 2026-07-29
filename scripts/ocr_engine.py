"""
OCR Engine Module for Document Processing.

Provides unified interface for multiple OCR backends:
- Tesseract (via pytesseract)
- RapidOCR (ONNX-based PaddleOCR architecture)
- EasyOCR (PyTorch-based OCR engine)

Outputs structured word & line level bounding boxes, text, and confidence scores.
"""

import os
import sys
import time
import json
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OCREngine")


def find_tesseract_cmd() -> Optional[str]:
    """Locate Tesseract executable on Windows or system PATH."""
    import shutil
    tess_in_path = shutil.which("tesseract")
    if tess_in_path:
        return tess_in_path

    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for p in common_paths:
        if os.path.exists(p):
            return p
    return None


class OCRResult:
    """Structured container for OCR extraction results."""
    def __init__(
        self,
        full_text: str,
        words: List[Dict[str, Any]],
        lines: List[Dict[str, Any]],
        engine_name: str,
        execution_time_sec: float
    ):
        self.full_text = full_text
        self.words = words  # list of {"text": str, "bbox": [xmin, ymin, xmax, ymax], "confidence": float}
        self.lines = lines  # list of {"text": str, "bbox": [xmin, ymin, xmax, ymax], "confidence": float}
        self.engine_name = engine_name
        self.execution_time_sec = execution_time_sec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "full_text": self.full_text,
            "word_count": len(self.words),
            "line_count": len(self.lines),
            "words": self.words,
            "lines": self.lines,
            "engine_name": self.engine_name,
            "execution_time_sec": round(self.execution_time_sec, 4),
        }


class BaseOCREngine:
    def extract(self, image_input: Union[str, np.ndarray, Image.Image]) -> OCRResult:
        raise NotImplementedError


class TesseractEngine(BaseOCREngine):
    def __init__(self, tesseract_cmd: Optional[str] = None, psm: int = 6, lang: str = "eng"):
        import pytesseract
        self.pytesseract = pytesseract
        cmd = tesseract_cmd or find_tesseract_cmd()
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
            logger.info(f"Initialized Tesseract engine with binary at: {cmd}")
        else:
            logger.warning("Tesseract binary not found in standard paths. pytesseract will rely on PATH.")
        self.psm = psm
        self.lang = lang

    def extract(self, image_input: Union[str, np.ndarray, Image.Image]) -> OCRResult:
        t0 = time.time()
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                image = Image.fromarray(image_input).convert("RGB")
            else:
                image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            image = image_input.convert("RGB")

        config = f"--psm {self.psm} -l {self.lang}"
        data = self.pytesseract.image_to_data(image, config=config, output_type=self.pytesseract.Output.DICT)

        words = []
        lines_dict: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text = str(data["text"][i]).strip()
            conf = float(data["conf"][i])
            if text and conf >= 0:
                xmin = data["left"][i]
                ymin = data["top"][i]
                xmax = xmin + data["width"][i]
                ymax = ymin + data["height"][i]
                bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]

                word_info = {"text": text, "bbox": bbox, "confidence": round(conf / 100.0, 4)}
                words.append(word_info)

                line_key = (data["page_num"][i], data["block_num"][i], data["line_num"][i])
                if line_key not in lines_dict:
                    lines_dict[line_key] = []
                lines_dict[line_key].append(word_info)

        lines = []
        full_text_lines = []
        for line_key, line_words in lines_dict.items():
            line_text = " ".join([w["text"] for w in line_words])
            if line_text.strip():
                xmin = min(w["bbox"][0] for w in line_words)
                ymin = min(w["bbox"][1] for w in line_words)
                xmax = max(w["bbox"][2] for w in line_words)
                ymax = max(w["bbox"][3] for w in line_words)
                avg_conf = sum(w["confidence"] for w in line_words) / len(line_words)
                lines.append({
                    "text": line_text,
                    "bbox": [xmin, ymin, xmax, ymax],
                    "confidence": round(avg_conf, 4)
                })
                full_text_lines.append(line_text)

        full_text = "\n".join(full_text_lines)
        t1 = time.time()
        return OCRResult(full_text, words, lines, "tesseract", t1 - t0)


class RapidOCREngine(BaseOCREngine):
    def __init__(self, max_side: int = 1280):
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.engine = RapidOCR()
            self.max_side = max_side
            logger.info("Initialized RapidOCR engine (ONNX runtime).")
        except ImportError:
            raise ImportError("rapidocr_onnxruntime is not installed. Install via: pip install rapidocr-onnxruntime")

    def extract(self, image_input: Union[str, np.ndarray, Image.Image]) -> OCRResult:
        t0 = time.time()
        if isinstance(image_input, str):
            image_np = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            image_np = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            image_np = image_input

        if image_np is None:
            raise ValueError(f"Could not load image input: {image_input}")

        h, w = image_np.shape[:2]
        max_dim = max(h, w)
        if self.max_side and max_dim > self.max_side:
            scale = float(self.max_side) / float(max_dim)
            new_w, new_h = int(w * scale), int(h * scale)
            ocr_input = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            ocr_input = image_np

        results, elapsed_time = self.engine(ocr_input)
        words = []
        lines = []
        full_text_lines = []

        if results:
            for item in results:
                # item format: [box_points, text, confidence]
                poly, text, conf = item[0], item[1], float(item[2])
                poly_np = np.array(poly) / scale
                xmin = int(np.min(poly_np[:, 0]))
                ymin = int(np.min(poly_np[:, 1]))
                xmax = int(np.max(poly_np[:, 0]))
                ymax = int(np.max(poly_np[:, 1]))
                bbox = [xmin, ymin, xmax, ymax]

                item_info = {
                    "text": text,
                    "bbox": bbox,
                    "confidence": round(conf, 4)
                }
                lines.append(item_info)
                full_text_lines.append(text)

                # Split line into pseudo-words for word-level granularity
                line_words = text.split()
                if line_words:
                    word_w = (xmax - xmin) / max(len(line_words), 1)
                    for idx, w in enumerate(line_words):
                        w_xmin = int(xmin + idx * word_w)
                        w_xmax = int(xmin + (idx + 1) * word_w)
                        words.append({
                            "text": w,
                            "bbox": [w_xmin, ymin, w_xmax, ymax],
                            "confidence": round(conf, 4)
                        })

        full_text = "\n".join(full_text_lines)
        t1 = time.time()
        return OCRResult(full_text, words, lines, "rapidocr", t1 - t0)


class EasyOCREngine(BaseOCREngine):
    def __init__(self, lang_list: List[str] = ["en"], gpu: bool = False):
        try:
            import easyocr
            self.reader = easyocr.Reader(lang_list, gpu=gpu)
            logger.info("Initialized EasyOCR engine.")
        except ImportError:
            raise ImportError("easyocr is not installed. Install via: pip install easyocr")

    def extract(self, image_input: Union[str, np.ndarray, Image.Image]) -> OCRResult:
        t0 = time.time()
        if isinstance(image_input, str):
            image_input_path_or_np = image_input
        elif isinstance(image_input, Image.Image):
            image_input_path_or_np = np.array(image_input)
        else:
            image_input_path_or_np = image_input

        results = self.reader.readtext(image_input_path_or_np)
        words = []
        lines = []
        full_text_lines = []

        if results:
            for item in results:
                # item: (bbox_corners, text, confidence)
                bbox_corners, text, conf = item[0], item[1], float(item[2])
                bbox_np = np.array(bbox_corners)
                xmin = int(np.min(bbox_np[:, 0]))
                ymin = int(np.min(bbox_np[:, 1]))
                xmax = int(np.max(bbox_np[:, 0]))
                ymax = int(np.max(bbox_np[:, 1]))
                bbox = [xmin, ymin, xmax, ymax]

                item_info = {
                    "text": text,
                    "bbox": bbox,
                    "confidence": round(conf, 4)
                }
                lines.append(item_info)
                full_text_lines.append(text)

                line_words = text.split()
                if line_words:
                    word_w = (xmax - xmin) / max(len(line_words), 1)
                    for idx, w in enumerate(line_words):
                        w_xmin = int(xmin + idx * word_w)
                        w_xmax = int(xmin + (idx + 1) * word_w)
                        words.append({
                            "text": w,
                            "bbox": [w_xmin, ymin, w_xmax, ymax],
                            "confidence": round(conf, 4)
                        })

        full_text = "\n".join(full_text_lines)
        t1 = time.time()
        return OCRResult(full_text, words, lines, "easyocr", t1 - t0)


def get_ocr_engine(name: str = "tesseract", **kwargs) -> BaseOCREngine:
    name_lower = name.lower()
    if name_lower == "tesseract":
        return TesseractEngine(**kwargs)
    elif name_lower == "rapidocr":
        return RapidOCREngine(**kwargs)
    elif name_lower == "easyocr":
        return EasyOCREngine(**kwargs)
    else:
        raise ValueError(f"Unknown OCR engine '{name}'. Choose from ['tesseract', 'rapidocr', 'easyocr'].")


def main():
    parser = argparse.ArgumentParser(description="Run OCR Extraction on receipt image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input receipt image")
    parser.add_argument("--engine", type=str, default="tesseract", choices=["tesseract", "rapidocr", "easyocr"], help="OCR engine name")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for JSON result")

    args = parser.parse_args()

    engine = get_ocr_engine(args.engine)
    logger.info(f"Extracting OCR text from '{args.image}' using {args.engine}...")
    result = engine.extract(args.image)

    print("\n--- Extracted Text ---")
    print(result.full_text)
    print("----------------------")
    print(f"Extracted {len(result.words)} words, {len(result.lines)} lines in {result.execution_time_sec:.3f} seconds.\n")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved OCR JSON result to {args.output}")


if __name__ == "__main__":
    main()
