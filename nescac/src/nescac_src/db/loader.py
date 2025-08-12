from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .session import get_session, init_db
from .models import Event, Meet, School, Athlete, ResultIndividual
from .mapping import parse_results_cell, normalize_result_entry, event_identity, to_seconds, normalize_school

# Default CSV path (repo_root/nescac/data/processed/clean/combined_individual_events.csv)
def _default_csv_path() -> Path:
    from pathlib import Path
    here = Path(__file__).resolve()
    repo = here.parents[4]
    return repo / "nescac" / "data" / "processed" / "clean" / "combined_individual_events.csv"


def _get_or_create(session, model, defaults=None, **kwargs):
    inst = session.execute(select(model).filter_by(**kwargs)).scalar_one_or_none()
    if inst:
        return inst
    params = dict(kwargs)
    if defaults:
        params.update(defaults)
    inst = model(**params)
    session.add(inst)
    try:
        session.flush()
    except IntegrityError:
        session.rollback()
        inst = session.execute(select(model).filter_by(**kwargs)).scalar_one()
    return inst


def load_combined_results(csv_path: Optional[Path] = None, chunk_size: int = 0) -> None:
    init_db()
    path = Path(csv_path or _default_csv_path())
    df = pd.read_csv(path)

    with get_session() as s:
        for _, row in df.iterrows():
            gender, distance, stroke, is_relay = event_identity(row)
            ev = _get_or_create(
                s,
                Event,
                gender=gender,
                distance=int(distance),
                stroke=stroke,
                is_relay=is_relay,
            )
            meet = _get_or_create(s, Meet, year=int(row["year"]), name=str(row.get("meet") or ""))

            res_list = parse_results_cell(row.get("results"))
            for r in res_list:
                r = normalize_result_entry(r)
                # Normalize school
                raw_school_name = (r.get("school") or "").strip() or None
                canonical_school_name = normalize_school(raw_school_name) if raw_school_name else None
                school = (
                    _get_or_create(s, School, name=canonical_school_name) if canonical_school_name else None
                )
                athlete = _get_or_create(
                    s,
                    Athlete,
                    name=str(r.get("name") or "").strip() or "Unknown",
                    school_id=school.id if school else None,
                )

                # Prelim row
                if r.get("prelim_time") is not None:
                    ri = ResultIndividual(
                        meet_id=meet.id,
                        event_id=ev.id,
                        athlete_id=athlete.id if athlete else None,
                        athlete_name=athlete.name if athlete else (r.get("name") or "Unknown"),
                        school_id=school.id if school else None,
                        school_name=canonical_school_name,
                        round="Prelim",
                        place=r.get("prelim_rank"),
                        time_seconds=r.get("prelim_seconds"),
                        time_raw=str(r.get("prelim_time")),
                    )
                    s.add(ri)
                    try:
                        s.flush()
                    except IntegrityError:
                        s.rollback()  # duplicate; skip

                # Finals row
                if r.get("finals_time") is not None:
                    ri = ResultIndividual(
                        meet_id=meet.id,
                        event_id=ev.id,
                        athlete_id=athlete.id if athlete else None,
                        athlete_name=athlete.name if athlete else (r.get("name") or "Unknown"),
                        school_id=school.id if school else None,
                        school_name=canonical_school_name,
                        round="Final",
                        place=r.get("final_rank"),
                        time_seconds=r.get("final_seconds"),
                        time_raw=str(r.get("finals_time")),
                    )
                    s.add(ri)
                    try:
                        s.flush()
                    except IntegrityError:
                        s.rollback()  # duplicate; skip


if __name__ == "__main__":
    load_combined_results()