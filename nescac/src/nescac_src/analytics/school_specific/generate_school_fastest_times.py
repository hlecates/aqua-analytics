import os
import re
import ast
from typing import Dict, List, Optional, Tuple

import pandas as pd


SCHOOLS: List[str] = [
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

# Case-insensitive regex patterns to recognize each school's variants
SCHOOL_PATTERNS: Dict[str, re.Pattern] = {
    "Amherst": re.compile(r"\bamherst\b", re.IGNORECASE),
    "Bates": re.compile(r"\bbates\b", re.IGNORECASE),
    "Bowdoin": re.compile(r"\bbowdoin\b", re.IGNORECASE),
    "Colby": re.compile(r"\bcolby\b", re.IGNORECASE),
    # Cover "Conn College", "Connecticut College"
    "Connecticut College": re.compile(r"\bconn(?:ecticut)?\s+college\b|\bconnecticut\s+college\b", re.IGNORECASE),
    "Hamilton": re.compile(r"\bhamilton\b", re.IGNORECASE),
    "Middlebury": re.compile(r"\bmiddlebury\b", re.IGNORECASE),
    "Trinity": re.compile(r"\btrinity\b", re.IGNORECASE),
    "Tufts": re.compile(r"\btufts\b", re.IGNORECASE),
    "Wesleyan": re.compile(r"\bwesleyan\b", re.IGNORECASE),
    "Williams": re.compile(r"\bwilliams\b", re.IGNORECASE),
}


def parse_time_to_seconds(time_str: Optional[str]) -> Optional[float]:

    if not time_str:
        return None

    # Remove non-time charcters eg colon and dot
    cleaned = re.sub(r"[^0-9:\.]", "", str(time_str))
    if cleaned == "":
        return None

    # Handle formats: MM:SS.ss or SS.ss
    if ":" in cleaned:
        try:
            minutes_str, seconds_str = cleaned.split(":", 1)
            return int(minutes_str) * 60 + float(seconds_str)
        except Exception:
            return None
    else:
        try:
            return float(cleaned)
        except Exception:
            return None


def detect_school(raw_school: Optional[str]) -> Optional[str]:
    if not raw_school:
        return None
    for name, pattern in SCHOOL_PATTERNS.items():
        if pattern.search(raw_school):
            return name
    return None


def get_swimmer_fastest_time(swimmer: Dict) -> Tuple[Optional[float], Optional[str]]:
    prelim_raw = swimmer.get("prelim_time")
    finals_raw = swimmer.get("finals_time")

    prelim_sec = parse_time_to_seconds(prelim_raw)
    finals_sec = parse_time_to_seconds(finals_raw)

    # Consider both if available, return minimum
    if prelim_sec is not None and finals_sec is not None:
        if finals_sec <= prelim_sec:
            return finals_sec, "Final"
        return prelim_sec, "Prelim"
    if finals_sec is not None:
        return finals_sec, "Final"
    if prelim_sec is not None:
        return prelim_sec, "Prelim"
    return None, None


def extract_fastest_by_school(
    input_csv_path: str,
    output_csv_path: str,
) -> pd.DataFrame:
    
    df = pd.read_csv(input_csv_path)

    # Prepare accumulator rows
    output_rows: List[Dict] = []

    # Iterate over rows 
    for _, row in df.iterrows():
        year = row.get("year")
        event_name = row.get("event_name")
        stroke = row.get("stroke")
        gender = row.get("gender")
        distance = row.get("distance")
        meet = row.get("meet")

        results_blob = row.get("results")
        if pd.isna(results_blob) or results_blob in (None, "", "[]"):
            continue

        # Parse the list[dict] safely
        try:
            swimmers: List[Dict] = ast.literal_eval(results_blob)
        except Exception:
            # If parsing fails, skip this row
            continue

        # For each canonical school, compute the fastest swimmer/time in this event
        # Filter swimmers per school first, then reduce to min
        school_to_best: Dict[str, Dict] = {}
        for swimmer in swimmers:
            raw_school = swimmer.get("school")
            canonical_school = detect_school(raw_school)
            if canonical_school is None:
                continue

            best_sec, round_label = get_swimmer_fastest_time(swimmer)
            if best_sec is None:
                continue

            # Track the best per school
            current_best = school_to_best.get(canonical_school)
            if current_best is None or best_sec < current_best["fastest_time_sec"]:
                school_to_best[canonical_school] = {
                    "year": year,
                    "event_name": event_name,
                    "stroke": stroke,
                    "gender": gender,
                    "distance": distance,
                    "meet": meet,
                    "school": canonical_school,
                    "swimmer_name": swimmer.get("name"),
                    "round": round_label,
                    "fastest_time_sec": best_sec,
                    # Preserve original display format from the specific round if available
                    "fastest_time_format": swimmer.get(
                        "finals_time" if round_label == "Final" else "prelim_time"
                    ),
                }

        # Emit rows for any schools that had participants in this event
        for _, best in school_to_best.items():
            output_rows.append(best)

    output_df = pd.DataFrame(output_rows)

    # Sort for readability
    if not output_df.empty:
        output_df = output_df.sort_values(
            by=["year", "gender", "event_name", "school", "fastest_time_sec"],
            ascending=[True, True, True, True, True],
            ignore_index=True,
        )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    output_df.to_csv(output_csv_path, index=False)

    return output_df


def main() -> None:
    project_root = "/Users/hlecates/Desktop/aqua-analytics"
    input_csv = os.path.join(
        project_root,
        "nescac",
        "data",
        "processed",
        "clean",
        "combined_individual_events.csv",
    )

    # Save under data/school-specific
    output_csv = os.path.join(
        project_root,
        "nescac",
        "data",
        "school-specific",
        "fastest_times_by_school.csv",
    )

    extract_fastest_by_school(input_csv, output_csv)


if __name__ == "__main__":
    main() 