"""Evaluator-only survival labels; never import this into the agent record tool."""
from datetime import timedelta
from nanoastra.oncology import timestamp, read_csv_gz

COLUMNS = ["patient_id", "death_time", "last_known_alive_time", "ascertainment_end_time"]


def horizon_label(row, index_time, horizon_days=180):
    if not isinstance(horizon_days, int) or horizon_days <= 0:
        raise ValueError("A positive integer horizon is required")
    index = timestamp(index_time)
    end = timestamp(row["ascertainment_end_time"])
    alive = timestamp(row["last_known_alive_time"])
    death = timestamp(row["death_time"]) if row["death_time"] else None
    if end < index or alive < index or alive > end:
        raise ValueError("Invalid outcome follow-up interval")
    if death and (death <= index or death > end or alive > death):
        raise ValueError("Invalid death/follow-up ordering")
    horizon = index + timedelta(days=horizon_days)
    observed_until = death or alive
    label = 1 if death and death <= horizon else (0 if observed_until >= horizon else None)
    return {"patient_id": row["patient_id"], "event_observed": int(death is not None),
            "followup_days": (observed_until-index).total_seconds()/86400,
            "horizon_days": horizon_days, "death_by_horizon": label}


def load_labels(path, index_time, horizon_days=180):
    rows = list(read_csv_gz(path, COLUMNS))
    if len({r["patient_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate outcome patient")
    return [horizon_label(r, index_time, horizon_days) for r in rows]
