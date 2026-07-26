from dataclasses import asdict, dataclass
import re
from typing import Any

import numpy as np
import pandas as pd

MISSING_TEXT = {"", "-", "—", "–", "/", "\\", "nan", "none", "n/a", "na"}
NUMBER = r"[-+]?\d+(?:\.\d+)?"


@dataclass(frozen=True)
class ParsedNumeric:
    value_raw: Any
    value_numeric: float | None
    unit_raw: str | None
    unit_standard: str | None
    range_lower: float | None
    range_upper: float | None
    parser_rule: str
    parser_version: str = "1.0"
    quality_flag: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_text(value: Any) -> str | float:
    if pd.isna(value):
        return np.nan
    text = re.sub(r"\s+", " ", str(value)).strip()
    return np.nan if text.lower() in MISSING_TEXT else text


def parse_numeric_range(value: Any, unit_standard: str | None = None) -> ParsedNumeric:
    cleaned = clean_text(value)
    if pd.isna(cleaned):
        return ParsedNumeric(value, None, None, unit_standard, None, None, "missing", quality_flag="missing")
    text = str(cleaned).replace(",", "")
    unit = re.sub(NUMBER, "", text).strip() or None
    range_match = re.search(rf"({NUMBER})\s*(?:-|~|–|—|至|到)\s*({NUMBER})", text)
    numbers = [float(item) for item in re.findall(NUMBER, text)]
    is_range = range_match is not None
    if range_match:
        endpoints = [float(range_match.group(1)), float(range_match.group(2))]
        lower, upper = min(endpoints), max(endpoints)
    else:
        lower, upper = (numbers[0], numbers[0]) if numbers else (None, None)
    numeric = (lower + upper) / 2 if lower is not None else None
    flag = "range_midpoint_for_model" if is_range else ("ok" if numbers else "unparsed")
    return ParsedNumeric(value, numeric, unit, unit_standard, lower, upper, "numeric_range", quality_flag=flag)


def parse_pvp_mw(value: Any) -> float:
    cleaned = clean_text(value)
    if pd.isna(cleaned):
        return np.nan
    text = str(cleaned).strip().replace(",", "")
    match = re.fullmatch(
        rf"\s*({NUMBER})\s*(万|kda|k|mda|m|da|g\s*/\s*mol|g·mol(?:-1|⁻¹)?)?\s*",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return np.nan
    number = float(match.group(1))
    unit = (match.group(2) or "da").lower().replace(" ", "")
    factors = {"万": 10000.0, "k": 1000.0, "kda": 1000.0, "m": 1_000_000.0, "mda": 1_000_000.0, "da": 1.0, "g/mol": 1.0, "g·mol-1": 1.0, "g·mol⁻¹": 1.0}
    return number * factors[unit]


def parse_ratio(value: Any) -> float:
    cleaned = clean_text(value)
    if pd.isna(cleaned):
        return np.nan
    numbers = [float(item) for item in re.findall(NUMBER, str(cleaned))]
    if len(numbers) >= 2 and re.search(r"[:：/]", str(cleaned)) and numbers[1] != 0:
        return numbers[0] / numbers[1]
    parsed = parse_numeric_range(value)
    return np.nan if parsed.value_numeric is None else parsed.value_numeric


def parse_duration_hours(value: Any) -> float:
    parsed = parse_numeric_range(value)
    if parsed.value_numeric is None:
        return np.nan
    return parsed.value_numeric / 60.0 if re.search(r"\bmin\b|分钟", str(value).lower()) else parsed.value_numeric
