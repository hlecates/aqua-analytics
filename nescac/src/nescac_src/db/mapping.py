import math
import re
from ast import literal_eval
from typing import Any, Dict, List, Optional, Tuple, Union


def to_seconds(x: Union[str, float, int, None]) -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.upper() in {"DQ", "DSQ", "NS", "NT"}:
        return None
    if ":" in s:
        mm, ss = s.split(":", 1)
        try:
            return int(mm) * 60 + float(ss)
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_results_cell(cell: Any) -> List[Dict[str, Any]]:
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return []
    if isinstance(cell, list):
        return cell
    try:
        parsed = literal_eval(cell)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def normalize_result_entry(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d)
    # normalize booleans
    out["exhibition"] = bool(out.get("exhibition", False))
    # ranks
    for k in ("prelim_rank", "final_rank"):
        v = out.get(k)
        try:
            out[k] = int(v) if v is not None and str(v).strip() != "" else None
        except Exception:
            out[k] = None
    # times
    out["prelim_seconds"] = to_seconds(out.get("prelim_time"))
    out["final_seconds"] = to_seconds(out.get("finals_time"))
    return out


# Canonical NESCAC schools
NESCAC_SCHOOLS = [
    "Amherst",
    "Bates",
    "Bowdoin",
    "Colby",
    "Connecticut College",
    "Hamilton",
    "Middlebury",
    "Trinity",
    "Tufts",
    "Wesleyan",
    "Williams",
]

# Precompiled regex patterns for school detection (case-insensitive)
_SCHOOL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bamherst\b", re.IGNORECASE), "Amherst"),
    (re.compile(r"\bbates\b", re.IGNORECASE), "Bates"),
    (re.compile(r"\bbowdoin\b", re.IGNORECASE), "Bowdoin"),
    (re.compile(r"\bcolby\b", re.IGNORECASE), "Colby"),
    # Be explicit for Connecticut College to avoid matching generic 'Conn.' state tags
    (re.compile(r"\bconnecticut\s+college\b", re.IGNORECASE), "Connecticut College"),
    (re.compile(r"\bconn\.?\s+college\b", re.IGNORECASE), "Connecticut College"),
    (re.compile(r"\bconnecticut\s+coll\b", re.IGNORECASE), "Connecticut College"),
    (re.compile(r"\bhamilton\b", re.IGNORECASE), "Hamilton"),
    (re.compile(r"\bmiddlebury\b", re.IGNORECASE), "Middlebury"),
    (re.compile(r"\btrinity\b", re.IGNORECASE), "Trinity"),
    (re.compile(r"\btufts\b", re.IGNORECASE), "Tufts"),
    (re.compile(r"\bwesleyan\b", re.IGNORECASE), "Wesleyan"),
    (re.compile(r"\bwilliams\b", re.IGNORECASE), "Williams"),
]


def normalize_school(raw: Union[str, None]) -> Optional[str]:
    """Detect and canonicalize a NESCAC school name from a raw string.

    Returns the canonical school name if a match is found; otherwise None.
    This strips extraneous tokens like class years, state abbreviations, and punctuation
    by matching known school name patterns within the string.
    """
    if not raw:
        return None
    text = str(raw)
    for pattern, canonical in _SCHOOL_PATTERNS:
        if pattern.search(text):
            return canonical
    return None


def event_identity(row) -> Tuple[str, int, str, bool]:
    # row has: gender, distance, stroke; dataset is individual events
    return str(row["gender"]), int(row["distance"]), str(row["stroke"]), False