"""
Field Extraction Module for Receipt Parser (SROIE Dataset).

Extracts 4 key fields from receipt text / OCR results:
- company: Store / Vendor Name
- date: Transaction Date
- address: Store / Merchant Address
- total: Total Amount Paid

Provides three extraction strategies:
1. RegexFieldExtractor: Pattern & regular expression matching.
2. LayoutHeuristicExtractor: Spatial line position, key-value anchors & geometry.
3. HybridFieldExtractor: Ensemble validator combining regex constraints & spatial heuristics.
"""

import os
import sys
import re
import json
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FieldExtractor")

# Known Malaysian state names & common address tokens
MALAYSIAN_STATES = [
    "JOHOR", "SELANGOR", "KUALA LUMPUR", "KL", "PENANG", "PULAU PINANG",
    "PERAK", "PAHANG", "MELAKA", "MALACCA", "SARAWAK", "SABAH",
    "NEGERI SEMBILAN", "KEDAH", "KELANTAN", "TERENGGANU", "LABUAN", "PUTRAJAYA", "WILAYAH PERSEKUTUAN"
]

COMPANY_SUFFIXES = [
    "SDN BHD", "SDN. BHD.", "SDN BHD.", "BHD", "BHD.", "ENTERPRISE", "TRADING",
    "RESTAURANT", "RESTORAN", "CAFE", "KOPITIAM", "MARKET", "SUPERMARKET",
    "MART", "PHARMACY", "FASHION", "STORE", "HOLDINGS", "CORP", "INC", "LIMITED",
    "S/B", "GIFT & HOME", "BAKERY", "STATIONERY", "HARDWARE", "SERVIS", "SERVICE"
]

ADDRESS_KEYWORDS = [
    "JALAN", "JLN", "TAMAN", "TMN", "NO.", "NO ", "LOT", "LEVEL", "FLOOR",
    "PERSIARAN", "BANDAR", "LEBUH", "COMPLEX", "KOMPLEKS", "PLAZA", "WISMA",
    "BLOCK", "BLOK", "SECTION", "SEKSYEN", "KAWASAN", "PASAR", "STREET", "ROAD",
    "BT ", "BATU ", "POSTCODE", "POSKOD"
]


class ExtractedFields:
    """Container for parsed receipt structured fields."""
    def __init__(
        self,
        company: str = "",
        date: str = "",
        address: str = "",
        total: str = "",
        confidences: Optional[Dict[str, float]] = None,
        extraction_method: str = "hybrid"
    ):
        self.company = company.strip()
        self.date = date.strip()
        self.address = address.strip()
        self.total = total.strip()
        self.confidences = confidences or {"company": 0.0, "date": 0.0, "address": 0.0, "total": 0.0}
        self.extraction_method = extraction_method

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company": self.company,
            "date": self.date,
            "address": self.address,
            "total": self.total,
            "confidences": self.confidences,
            "extraction_method": self.extraction_method
        }


def clean_text_line(line: str) -> str:
    """Clean raw OCR line string by fixing spaces around punctuation."""
    text = line.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_amount(amount_str: str) -> str:
    """Normalize extracted monetary string into a standard decimal format (e.g. 12.34)."""
    clean = re.sub(r'[^\d\.]', '', amount_str)
    # Handle cases with multiple dots (e.g., 12.34.5 -> 12.34)
    parts = clean.split('.')
    if len(parts) > 2:
        clean = parts[0] + '.' + ''.join(parts[1:])
    try:
        val = float(clean)
        return f"{val:.2f}"
    except ValueError:
        return amount_str


class BaseFieldExtractor:
    def extract_fields(self, ocr_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> ExtractedFields:
        raise NotImplementedError


class RegexFieldExtractor(BaseFieldExtractor):
    """Regex & Pattern-Based Field Extractor."""

    def __init__(self):
        # Date regex patterns
        self.date_patterns = [
            # DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, YYYY/MM/DD, YYYY-MM-DD
            r'\b(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})\b',
            # YYYY/MM/DD or YYYY-MM-DD
            r'\b(\d{4}[/\.-]\d{1,2}[/\.-]\d{1,2})\b',
            # DD MMM YYYY (e.g., 25 DEC 2018 or 25-DEC-2018)
            r'\b(\d{1,2}[-\s]+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[a-z]*[-\s]+\d{2,4})\b',
        ]

        # Total amount regex patterns
        self.total_patterns = [
            r'(?:TOTAL|AMOUNT|NETT|NET|GRAND\s+TOTAL|TOTAL\s+AMOUNT|TOTAL\s+RM|RM)\s*[:\.\=]?\s*(?:RM)?\s*\$?(\d+\.\d{2})\b',
            r'(?:TOTAL|AMOUNT|NETT|GRAND\s+TOTAL)\s*[:\.\=]?\s*(?:RM)?\s*\$?(\d+[\.,]\d{2})\b',
            r'\b(?:RM|\$)\s*(\d+\.\d{2})\b',
        ]

    def extract_fields(self, ocr_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> ExtractedFields:
        if isinstance(ocr_data, str):
            lines = [clean_text_line(l) for l in ocr_data.split('\n') if l.strip()]
            full_text = ocr_data
        elif isinstance(ocr_data, dict):
            lines = [clean_text_line(l.get("text", "")) for l in ocr_data.get("lines", []) if l.get("text", "").strip()]
            full_text = ocr_data.get("full_text", "\n".join(lines))
        elif isinstance(ocr_data, list):
            lines = [clean_text_line(l.get("text", "")) for l in ocr_data if l.get("text", "").strip()]
            full_text = "\n".join(lines)
        else:
            lines = []
            full_text = ""

        company = self._extract_company(lines)
        date = self._extract_date(full_text)
        address = self._extract_address(lines, company)
        total = self._extract_total(lines, full_text)

        conf = {
            "company": 0.7 if company else 0.0,
            "date": 0.85 if date else 0.0,
            "address": 0.65 if address else 0.0,
            "total": 0.75 if total else 0.0
        }

        return ExtractedFields(
            company=company,
            date=date,
            address=address,
            total=total,
            confidences=conf,
            extraction_method="regex"
        )

    def _extract_date(self, full_text: str) -> str:
        for pat in self.date_patterns:
            match = re.search(pat, full_text, re.IGNORECASE)
            if match:
                date_str = match.group(1).strip()
                # Clean up date punctuation spacing
                date_str = re.sub(r'\s+', '', date_str)
                return date_str
        return ""

    def _extract_total(self, lines: List[str], full_text: str) -> str:
        # Check explicit keywords line by line in reverse (since total is usually near bottom)
        for line in reversed(lines):
            line_upper = line.upper()
            if any(k in line_upper for k in ["TOTAL", "NETT", "AMOUNT DUE", "TOTAL RM", "GRAND TOTAL"]):
                if "SUBTOTAL" in line_upper or "CHANGE" in line_upper or "ROUNDING" in line_upper or "GST" in line_upper:
                    continue
                amounts = re.findall(r'\d+\.\d{2}', line)
                if amounts:
                    return normalize_amount(amounts[-1])

        # Regex pattern matching across full text
        for pat in self.total_patterns:
            matches = re.findall(pat, full_text, re.IGNORECASE)
            if matches:
                return normalize_amount(matches[-1])

        # Fallback: largest monetary number found in bottom half of receipt
        bottom_lines = lines[len(lines) // 2:] if len(lines) > 2 else lines
        all_amounts = []
        for line in bottom_lines:
            if any(k in line.upper() for k in ["CARD", "CASH", "CHANGE", "SUBTOTAL", "TAX"]):
                continue
            found = re.findall(r'\b\d+\.\d{2}\b', line)
            for f in found:
                try:
                    all_amounts.append(float(f))
                except ValueError:
                    pass
        if all_amounts:
            return f"{max(all_amounts):.2f}"

        return ""

    def _extract_company(self, lines: List[str]) -> str:
        if not lines:
            return ""

        # Candidates from top 5 lines
        top_candidates = lines[:min(5, len(lines))]

        # High priority: line containing company suffixes
        for line in top_candidates:
            line_upper = line.upper()
            if any(s in line_upper for s in COMPANY_SUFFIXES):
                # Ensure line is not an address line
                if not any(a in line_upper for a in ADDRESS_KEYWORDS) and not re.search(r'\b\d{5}\b', line):
                    return line

        # Fallback: first non-empty line that isn't a date, phone number, GST/ROC, or address
        for line in top_candidates:
            line_upper = line.upper()
            if re.search(r'\b\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4}\b', line):
                continue
            if re.search(r'\b(TEL|FAX|PHONE|GST|ROC|REG|NO\.|DATE|RECEIPT|INVOICE)\b', line_upper):
                continue
            if any(a in line_upper for a in ADDRESS_KEYWORDS) or re.search(r'\b\d{5}\b', line):
                continue
            if len(line) >= 3 and re.search(r'[A-Za-z]', line):
                return line

        return top_candidates[0] if top_candidates else ""

    def _extract_address(self, lines: List[str], company_name: str) -> str:
        if not lines:
            return ""

        address_lines = []
        in_address_block = False

        for idx, line in enumerate(lines[:12]):  # Address is almost always in top 12 lines
            if company_name and line == company_name:
                continue

            line_upper = line.upper()

            # Stop address parsing if line contains invoice/receipt metadata
            if re.search(r'\b(TEL|FAX|PHONE|GST|ROC|REG|DATE|RECEIPT|INVOICE|CASHIER|TABLE)\b', line_upper):
                if in_address_block:
                    break
                else:
                    continue

            has_address_kw = any(kw in line_upper for kw in ADDRESS_KEYWORDS)
            has_state = any(st in line_upper for st in MALAYSIAN_STATES)
            has_postcode = bool(re.search(r'\b\d{5}\b', line))

            if has_address_kw or has_state or has_postcode or in_address_block:
                # Add line if it looks like an address component
                if len(line) > 3 and not re.search(r'\b\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4}\b', line):
                    address_lines.append(line)
                    in_address_block = True
                    if has_state or has_postcode:
                        break  # State or postcode usually marks the end of the address block

        if address_lines:
            full_addr = ", ".join(address_lines)
            # Remove double commas or extra spaces
            full_addr = re.sub(r'\s*,\s*', ', ', full_addr)
            full_addr = re.sub(r',+', ',', full_addr)
            return full_addr

        return ""


class LayoutHeuristicExtractor(BaseFieldExtractor):
    """Spatial Layout & Candidate Scoring Field Extractor."""

    def extract_fields(self, ocr_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> ExtractedFields:
        if isinstance(ocr_data, dict):
            line_objs = ocr_data.get("lines", [])
            full_text = ocr_data.get("full_text", "")
        elif isinstance(ocr_data, list):
            line_objs = ocr_data
            full_text = "\n".join([l.get("text", "") for l in line_objs])
        elif isinstance(ocr_data, str):
            lines_str = [l.strip() for l in ocr_data.split('\n') if l.strip()]
            line_objs = [{"text": l, "bbox": [0, idx * 20, 500, (idx + 1) * 20]} for idx, l in enumerate(lines_str)]
            full_text = ocr_data
        else:
            line_objs = []
            full_text = ""

        # Sort line objects top to bottom by ymin coordinate
        line_objs = sorted(line_objs, key=lambda x: x.get("bbox", [0, 0, 0, 0])[1])
        line_texts = [clean_text_line(l.get("text", "")) for l in line_objs if l.get("text", "").strip()]

        company = self._extract_company_spatial(line_objs)
        date = self._extract_date_spatial(line_objs, full_text)
        address = self._extract_address_spatial(line_objs, company)
        total = self._extract_total_spatial(line_objs)

        conf = {
            "company": 0.75 if company else 0.0,
            "date": 0.85 if date else 0.0,
            "address": 0.70 if address else 0.0,
            "total": 0.80 if total else 0.0
        }

        return ExtractedFields(
            company=company,
            date=date,
            address=address,
            total=total,
            confidences=conf,
            extraction_method="layout_heuristic"
        )

    def _extract_company_spatial(self, line_objs: List[Dict[str, Any]]) -> str:
        if not line_objs:
            return ""

        candidates = []
        for idx, obj in enumerate(line_objs[:6]):
            text = clean_text_line(obj.get("text", ""))
            if not text or len(text) < 2:
                continue

            text_upper = text.upper()
            score = 10.0 - idx * 1.5  # Header position preference

            # Company name indicators
            if any(s in text_upper for s in COMPANY_SUFFIXES):
                score += 15.0
            if re.search(r'\b(SDN|BHD|ENTERPRISE|LIMITED|TRADING)\b', text_upper):
                score += 10.0

            # Penalties
            if re.search(r'\b\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4}\b', text):
                score -= 20.0
            if any(a in text_upper for a in ADDRESS_KEYWORDS) or re.search(r'\b\d{5}\b', text):
                score -= 15.0
            if re.search(r'\b(TEL|FAX|PHONE|GST|ROC|TAX|REG|INVOICE|CASHIER|WELCOME)\b', text_upper):
                score -= 15.0

            candidates.append((score, text))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            if candidates[0][0] > 0:
                return candidates[0][1]

        return line_objs[0].get("text", "").strip() if line_objs else ""

    def _extract_date_spatial(self, line_objs: List[Dict[str, Any]], full_text: str) -> str:
        # Search lines near "DATE", "TIME", or top lines first
        for obj in line_objs:
            text = obj.get("text", "")
            match = re.search(r'\b(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})\b', text)
            if match:
                return re.sub(r'\s+', '', match.group(1))

        # Full text fallback search
        match = re.search(r'\b(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})\b', full_text)
        if match:
            return re.sub(r'\s+', '', match.group(1))

        match_month = re.search(r'\b(\d{1,2}[-\s]+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[a-z]*[-\s]+\d{2,4})\b', full_text, re.IGNORECASE)
        if match_month:
            return match_month.group(1).strip()

        return ""

    def _extract_address_spatial(self, line_objs: List[Dict[str, Any]], company_name: str) -> str:
        address_tokens = []
        recording = False

        for obj in line_objs[:14]:
            text = clean_text_line(obj.get("text", ""))
            if not text or text == company_name:
                continue

            text_upper = text.upper()
            if re.search(r'\b(TEL|FAX|PHONE|GST|ROC|REG|DATE|RECEIPT|INVOICE|CASHIER)\b', text_upper):
                if recording:
                    break
                continue

            is_addr_line = (
                any(kw in text_upper for kw in ADDRESS_KEYWORDS) or
                any(st in text_upper for st in MALAYSIAN_STATES) or
                bool(re.search(r'\b\d{5}\b', text))
            )

            if is_addr_line or recording:
                if len(text) > 3 and not re.search(r'\b\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4}\b', text):
                    address_tokens.append(text)
                    recording = True
                    if any(st in text_upper for st in MALAYSIAN_STATES) or re.search(r'\b\d{5}\b', text):
                        break

        if address_tokens:
            addr = ", ".join(address_tokens)
            addr = re.sub(r'\s*,\s*', ', ', addr)
            return addr

        return ""

    def _extract_total_spatial(self, line_objs: List[Dict[str, Any]]) -> str:
        candidates = []

        for idx, obj in enumerate(line_objs):
            text = clean_text_line(obj.get("text", ""))
            text_upper = text.upper()

            # Skip header lines, subtotal, change, rounding, tax
            if any(k in text_upper for k in ["SUBTOTAL", "SUB-TOTAL", "CHANGE", "ROUNDING", "TAX", "GST"]):
                continue

            amounts = re.findall(r'\b\d+\.\d{2}\b', text)
            if not amounts:
                continue

            val_str = amounts[-1]
            try:
                val = float(val_str)
            except ValueError:
                continue

            score = 0.0
            if "TOTAL" in text_upper:
                score += 50.0
            if "GRAND TOTAL" in text_upper or "NETT" in text_upper or "NET TOTAL" in text_upper:
                score += 60.0
            if "AMOUNT" in text_upper or "AMOUNT DUE" in text_upper:
                score += 40.0
            if "RM" in text_upper or "$" in text:
                score += 20.0

            # Bottom position bias
            rel_pos = float(idx) / float(max(len(line_objs), 1))
            score += rel_pos * 30.0

            candidates.append((score, val, val_str))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return f"{candidates[0][1]:.2f}"

        return ""


class HybridFieldExtractor(BaseFieldExtractor):
    """Ensemble Hybrid Field Extractor combining Regex & Layout Heuristics."""

    def __init__(self):
        self.regex_extractor = RegexFieldExtractor()
        self.layout_extractor = LayoutHeuristicExtractor()

    def extract_fields(self, ocr_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> ExtractedFields:
        regex_res = self.regex_extractor.extract_fields(ocr_data)
        layout_res = self.layout_extractor.extract_fields(ocr_data)

        # Ensemble resolution
        company = self._select_company(regex_res.company, layout_res.company)
        date = self._select_date(regex_res.date, layout_res.date)
        address = self._select_address(regex_res.address, layout_res.address)
        total = self._select_total(regex_res.total, layout_res.total)

        conf = {
            "company": 0.85 if company else 0.0,
            "date": 0.90 if date else 0.0,
            "address": 0.78 if address else 0.0,
            "total": 0.88 if total else 0.0
        }

        return ExtractedFields(
            company=company,
            date=date,
            address=address,
            total=total,
            confidences=conf,
            extraction_method="hybrid"
        )

    def _select_company(self, regex_company: str, layout_company: str) -> str:
        if not regex_company:
            return layout_company
        if not layout_company:
            return regex_company

        # Prefer candidate with explicit SDN BHD / corporate suffix
        regex_has_suffix = any(s in regex_company.upper() for s in COMPANY_SUFFIXES)
        layout_has_suffix = any(s in layout_company.upper() for s in COMPANY_SUFFIXES)

        if regex_has_suffix and not layout_has_suffix:
            return regex_company
        if layout_has_suffix and not regex_has_suffix:
            return layout_company

        return regex_company if len(regex_company) >= len(layout_company) else layout_company

    def _select_date(self, regex_date: str, layout_date: str) -> str:
        if regex_date and re.search(r'\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4}', regex_date):
            return regex_date
        if layout_date and re.search(r'\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4}', layout_date):
            return layout_date
        return regex_date or layout_date

    def _select_address(self, regex_address: str, layout_address: str) -> str:
        if not regex_address:
            return layout_address
        if not layout_address:
            return regex_address
        # Prefer the longer address with Malaysian state or postcode
        reg_score = len(regex_address) + (20 if any(s in regex_address.upper() for s in MALAYSIAN_STATES) else 0)
        lay_score = len(layout_address) + (20 if any(s in layout_address.upper() for s in MALAYSIAN_STATES) else 0)
        return regex_address if reg_score >= lay_score else layout_address

    def _select_total(self, regex_total: str, layout_total: str) -> str:
        if regex_total and layout_total and regex_total == layout_total:
            return regex_total
        if regex_total:
            return regex_total
        return layout_total


def get_field_extractor(method: str = "hybrid") -> BaseFieldExtractor:
    m = method.lower()
    if m == "regex":
        return RegexFieldExtractor()
    elif m == "layout" or m == "layout_heuristic":
        return LayoutHeuristicExtractor()
    elif m == "hybrid":
        return HybridFieldExtractor()
    else:
        raise ValueError(f"Unknown field extraction method '{method}'. Choose from ['regex', 'layout', 'hybrid'].")


def main():
    parser = argparse.ArgumentParser(description="Extract structured fields (company, date, address, total) from receipt.")
    parser.add_argument("--image", type=str, default=None, help="Path to receipt image file")
    parser.add_argument("--ocr-json", type=str, default=None, help="Path to precomputed OCR JSON file")
    parser.add_argument("--method", type=str, default="hybrid", choices=["regex", "layout", "hybrid"], help="Extraction strategy")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for extracted fields JSON")
    args = parser.parse_args()

    if not args.image and not args.ocr_json:
        parser.error("Either --image or --ocr-json must be provided!")

    if args.ocr_json:
        with open(args.ocr_json, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)
    else:
        from ocr_engine import get_ocr_engine
        engine = get_ocr_engine("rapidocr")
        ocr_result = engine.extract(args.image)
        ocr_data = ocr_result.to_dict()

    extractor = get_field_extractor(args.method)
    fields = extractor.extract_fields(ocr_data)

    print("\n" + "=" * 50)
    print(f"  EXTRACTED RECEIPT FIELDS ({args.method.upper()})")
    print("=" * 50)
    print(f"Company:  {fields.company}")
    print(f"Date:     {fields.date}")
    print(f"Address:  {fields.address}")
    print(f"Total:    {fields.total}")
    print("=" * 50 + "\n")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(fields.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved extracted fields JSON to {args.output}")


if __name__ == "__main__":
    main()
