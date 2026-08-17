"""
Validation & Confidence Scoring Module for Receipt Parser (SROIE Dataset).

Provides rule-based sanity checks, per-field dynamic confidence scoring,
and low-confidence extraction flagging for receipt fields (company, date, address, total).
"""

import re
import datetime
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ReceiptValidator")

# Known Malaysian state names & address keywords
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

BLACK_LIST_COMPANY_KEYWORDS = [
    "TEL:", "FAX:", "PHONE", "GST", "ROC", "TAX INVOICE", "CASH RECEIPT",
    "RECEIPT", "INVOICE", "WELCOME", "THANK YOU", "TOTAL", "SUBTOTAL"
]


class ValidationResult:
    """Container for validated receipt field results, scores, and warning flags."""
    def __init__(
        self,
        fields: Dict[str, str],
        confidences: Dict[str, float],
        overall_confidence: float,
        is_valid: bool,
        requires_review: bool,
        flags: List[str],
        iso_date: Optional[str] = None,
        numeric_total: Optional[float] = None
    ):
        self.fields = fields
        self.confidences = confidences
        self.overall_confidence = round(overall_confidence, 4)
        self.is_valid = is_valid
        self.requires_review = requires_review
        self.flags = flags
        self.iso_date = iso_date
        self.numeric_total = numeric_total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fields": self.fields,
            "confidences": self.confidences,
            "overall_confidence": self.overall_confidence,
            "is_valid": self.is_valid,
            "requires_review": self.requires_review,
            "flags": self.flags,
            "iso_date": self.iso_date,
            "numeric_total": self.numeric_total
        }


class ReceiptValidator:
    """Rule-based validator and dynamic confidence scorer for extracted receipt fields."""

    def __init__(self, confidence_threshold: float = 0.70):
        self.confidence_threshold = confidence_threshold

    def validate_date(self, date_str: str) -> Tuple[bool, Optional[str], float, List[str]]:
        """
        Validate date format, check calendar sanity, convert to ISO YYYY-MM-DD.
        Returns: (is_valid, iso_date_str, confidence_score, flags)
        """
        flags = []
        if not date_str or not date_str.strip():
            flags.append("MISSING_FIELD:date")
            return False, None, 0.0, flags

        clean_date = date_str.strip()
        parsed_dt = None
        conf = 0.50

        # Pattern 1: DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
        m1 = re.match(r'^(\d{1,2})[/\.-](\d{1,2})[/\.-](\d{2,4})$', clean_date)
        if m1:
            d, m, y = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
            if y < 100:
                y += 2000 if y < 50 else 1900
            try:
                parsed_dt = datetime.date(y, m, d)
                conf = 0.90
            except ValueError:
                # Try MM/DD/YYYY if DD/MM/YYYY failed
                try:
                    parsed_dt = datetime.date(y, d, m)
                    conf = 0.80
                except ValueError:
                    flags.append("INVALID_CALENDAR_DATE:date")

        # Pattern 2: YYYY/MM/DD or YYYY-MM-DD
        if not parsed_dt:
            m2 = re.match(r'^(\d{4})[/\.-](\d{1,2})[/\.-](\d{1,2})$', clean_date)
            if m2:
                y, m, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
                try:
                    parsed_dt = datetime.date(y, m, d)
                    conf = 0.95
                except ValueError:
                    flags.append("INVALID_CALENDAR_DATE:date")

        # Pattern 3: DD MMM YYYY (e.g., 25 DEC 2018 or 25-DEC-2018)
        if not parsed_dt:
            m3 = re.match(r'^(\d{1,2})[-\s]+([A-Za-z]{3,9})[-\s]+(\d{2,4})$', clean_date)
            if m3:
                d_str, m_str, y_str = m3.group(1), m3.group(2), m3.group(3)
                y = int(y_str)
                if y < 100:
                    y += 2000 if y < 50 else 1900
                months_map = {
                    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
                }
                m_code = m_str[:3].upper()
                if m_code in months_map:
                    try:
                        parsed_dt = datetime.date(y, months_map[m_code], int(d_str))
                        conf = 0.95
                    except ValueError:
                        flags.append("INVALID_CALENDAR_DATE:date")

        if not parsed_dt:
            flags.append("UNRECOGNIZED_DATE_FORMAT:date")
            return False, None, round(conf, 4), flags

        # Calendar year sanity check (1990 - 2027)
        if parsed_dt.year < 1990 or parsed_dt.year > 2027:
            flags.append("OUT_OF_RANGE_YEAR:date")
            conf -= 0.30

        # Check if date is far in future
        if parsed_dt > datetime.date.today() + datetime.timedelta(days=365):
            flags.append("FUTURE_DATE_WARNING:date")
            conf -= 0.20

        iso_str = parsed_dt.isoformat()
        is_valid = len(flags) == 0 or all("WARNING" in f for f in flags)
        return is_valid, iso_str, round(max(0.0, min(1.0, conf)), 4), flags

    def validate_total(self, total_str: str, raw_text: str = "") -> Tuple[bool, Optional[float], float, List[str]]:
        """
        Validate total monetary amount string, range sanity ($0.01 - $50,000), keyword proximity.
        Returns: (is_valid, numeric_total, confidence_score, flags)
        """
        flags = []
        if not total_str or not total_str.strip():
            flags.append("MISSING_FIELD:total")
            return False, None, 0.0, flags

        clean = re.sub(r'[^\d\.]', '', total_str.strip())
        parts = clean.split('.')
        if len(parts) > 2:
            clean = parts[0] + '.' + ''.join(parts[1:])

        try:
            val = float(clean)
        except ValueError:
            flags.append("NON_NUMERIC_TOTAL:total")
            return False, None, 0.0, flags

        conf = 0.60

        # Range Sanity Checks
        if val <= 0.0:
            flags.append("ZERO_OR_NEGATIVE_TOTAL:total")
            conf -= 0.50
        elif val > 50000.0:
            flags.append("UNUSUALLY_HIGH_TOTAL:total")
            conf -= 0.30
        else:
            conf += 0.15

        # Check decimal places (standard receipts have 2 decimal places)
        if '.' in total_str:
            dec_len = len(total_str.split('.')[-1])
            if dec_len == 2:
                conf += 0.15
            else:
                flags.append("NON_STANDARD_DECIMAL_TOTAL:total")

        # Keyword proximity check if raw_text provided
        if raw_text:
            text_upper = raw_text.upper()
            if any(k in text_upper for k in ["TOTAL", "NETT", "GRAND TOTAL", "AMOUNT DUE", "RM"]):
                conf += 0.10

        is_valid = val > 0.0 and val <= 50000.0
        return is_valid, round(val, 2), round(max(0.0, min(1.0, conf)), 4), flags

    def validate_company(self, company_str: str) -> Tuple[bool, float, List[str]]:
        """
        Validate vendor/store name, string length, keyword blacklist, corporate suffix.
        Returns: (is_valid, confidence_score, flags)
        """
        flags = []
        if not company_str or not company_str.strip():
            flags.append("MISSING_FIELD:company")
            return False, 0.0, flags

        clean = company_str.strip()
        clean_upper = clean.upper()
        conf = 0.50

        # Minimum length check
        if len(clean) < 3:
            flags.append("TOO_SHORT_COMPANY_NAME:company")
            conf -= 0.30
        else:
            conf += 0.15

        # Blacklisted keyword contamination
        for blk in BLACK_LIST_COMPANY_KEYWORDS:
            if blk in clean_upper:
                flags.append(f"CONTAMINATED_COMPANY_NAME:{blk}")
                conf -= 0.40

        # Known corporate suffix boost
        if any(s in clean_upper for s in COMPANY_SUFFIXES):
            conf += 0.25

        # Must contain letters
        if not re.search(r'[A-Za-z]', clean):
            flags.append("NO_LETTERS_IN_COMPANY_NAME:company")
            conf -= 0.40

        is_valid = len(clean) >= 3 and re.search(r'[A-Za-z]', clean) and conf >= 0.50
        return is_valid, round(max(0.0, min(1.0, conf)), 4), flags

    def validate_address(self, address_str: str) -> Tuple[bool, float, List[str]]:
        """
        Validate address string, check street keywords, Malaysian state, postcode.
        Returns: (is_valid, confidence_score, flags)
        """
        flags = []
        if not address_str or not address_str.strip():
            flags.append("MISSING_FIELD:address")
            return False, 0.0, flags

        clean = address_str.strip()
        clean_upper = clean.upper()
        conf = 0.50

        # Check address keywords
        has_kw = any(kw in clean_upper for kw in ADDRESS_KEYWORDS)
        if has_kw:
            conf += 0.20

        # Check Malaysian state
        has_state = any(st in clean_upper for st in MALAYSIAN_STATES)
        if has_state:
            conf += 0.15

        # Check 5-digit postcode
        has_postcode = bool(re.search(r'\b\d{5}\b', clean))
        if has_postcode:
            conf += 0.15

        # Length check
        if len(clean) < 10:
            flags.append("TOO_SHORT_ADDRESS:address")
            conf -= 0.20

        if not (has_kw or has_state or has_postcode):
            flags.append("WEAK_ADDRESS_STRUCTURE:address")

        is_valid = len(clean) >= 10 and (has_kw or has_state or has_postcode)
        return is_valid, round(max(0.0, min(1.0, conf)), 4), flags

    def validate_fields(
        self,
        fields_dict: Dict[str, str],
        raw_text: str = "",
        method_confidences: Optional[Dict[str, float]] = None
    ) -> ValidationResult:
        """
        Perform comprehensive validation across all 4 receipt fields,
        computing per-field dynamic confidence scores, overall score, and flags.
        """
        all_flags = []

        # Validate Date
        date_val = fields_dict.get("date", "")
        d_valid, iso_date, d_conf, d_flags = self.validate_date(date_val)
        all_flags.extend(d_flags)

        # Validate Total
        total_val = fields_dict.get("total", "")
        t_valid, num_total, t_conf, t_flags = self.validate_total(total_val, raw_text=raw_text)
        all_flags.extend(t_flags)

        # Validate Company
        comp_val = fields_dict.get("company", "")
        c_valid, c_conf, c_flags = self.validate_company(comp_val)
        all_flags.extend(c_flags)

        # Validate Address
        addr_val = fields_dict.get("address", "")
        a_valid, a_conf, a_flags = self.validate_address(addr_val)
        all_flags.extend(a_flags)

        # Blend with method confidences if provided
        if method_confidences:
            c_conf = round(0.5 * c_conf + 0.5 * method_confidences.get("company", 0.7), 4)
            d_conf = round(0.5 * d_conf + 0.5 * method_confidences.get("date", 0.7), 4)
            a_conf = round(0.5 * a_conf + 0.5 * method_confidences.get("address", 0.7), 4)
            t_conf = round(0.5 * t_conf + 0.5 * method_confidences.get("total", 0.7), 4)

        confidences = {
            "company": c_conf,
            "date": d_conf,
            "address": a_conf,
            "total": t_conf
        }

        # Check low-confidence extractions against threshold
        for field, conf_val in confidences.items():
            if conf_val < self.confidence_threshold:
                all_flags.append(f"LOW_CONFIDENCE:{field}")

        # Overall receipt score (weighted average: Total & Date get 30% weight each, Company & Address get 20% each)
        overall_conf = 0.30 * t_conf + 0.30 * d_conf + 0.20 * c_conf + 0.20 * a_conf

        requires_review = overall_conf < self.confidence_threshold or len(all_flags) > 0
        is_valid = d_valid and t_valid and c_valid and not requires_review

        return ValidationResult(
            fields=fields_dict,
            confidences=confidences,
            overall_confidence=overall_conf,
            is_valid=is_valid,
            requires_review=requires_review,
            flags=all_flags,
            iso_date=iso_date,
            numeric_total=num_total
        )
