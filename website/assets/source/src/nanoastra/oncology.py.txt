"""Point-in-time projection of explicitly synthetic oncology CSV exports.

Domain-specific ingestion only: verl remains the RL infrastructure. Outcome
loading lives in a separate module and is never called by this projection.
"""
import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "fixtures/oncology/clinical"
INDEX_TIME = "2025-03-01T09:00:00Z"
TABLES = {
    "patients": ["age_at_index", "sex", "birth_year"],
    "encounters": ["encounter_type", "department", "reason"],
    "diagnosis": ["code_system", "code", "description", "verification_status"],
    "labs": ["specimen_id", "test_code", "test_name", "value_num", "value_text", "unit", "ref_low", "ref_high"],
    "vitals": ["measurement", "value_num", "unit", "context"],
    "notes": ["note_type", "author_role", "text"],
    "pathology": ["specimen_site", "histology", "report_text"],
    "staging": ["stage_system", "stage_group", "basis", "disease_sites"],
    "biomarkers": ["marker", "assay", "result", "result_status"],
    "medication_orders": ["drug", "intent", "order_status"],
    "medication_administrations": ["drug", "dose", "unit", "route", "administration_status"],
    "procedures": ["procedure_name", "procedure_status", "result_summary"],
    "performance_status": ["scale", "score", "assessor", "context"],
}
COMMON = ["record_id", "version", "patient_id", "encounter_id", "event_time", "available_time", "record_status", "source_system"]


def timestamp(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Timezone-aware timestamps are required")
    return dt.astimezone(timezone.utc)


def read_csv_gz(path, required):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != required:
            raise ValueError(f"Unexpected schema in {Path(path).name}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row in {Path(path).name}")
            yield row


def snapshot(data_dir=DEFAULT_DATA, patient_id="SYN-ONC-001", index_time=INDEX_TIME):
    """Return only evidence observable by the index time, with row provenance.

    Missing/invalid clocks fail closed. Duplicate versions and inverted time
    ordering fail validation. Future rows are never returned, including counts.
    Latest available version wins before cancellation filtering.
    """
    cutoff = timestamp(index_time)
    result = {}
    for table, extra in TABLES.items():
        latest, seen = {}, set()
        for row in read_csv_gz(Path(data_dir) / f"{table}.csv.gz", COMMON + extra):
            key = (row["patient_id"], row["record_id"], row["version"])
            if key in seen:
                raise ValueError(f"Duplicate record version in {table}")
            seen.add(key)
            if not row["record_id"] or not row["patient_id"]:
                raise ValueError("Missing record or patient identifier")
            if row["record_status"] not in {"active", "cancelled"}:
                raise ValueError("Unknown record status")
            try:
                event, available = timestamp(row["event_time"]), timestamp(row["available_time"])
                version = int(row["version"])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid clock/version in {table}:{row['record_id']}") from None
            if version < 1 or available < event:
                raise ValueError("Invalid event/availability ordering or version")
            if row["patient_id"] != patient_id or event > cutoff or available > cutoff:
                continue
            rid = row["record_id"]
            if rid not in latest or (available, version) > latest[rid][0]:
                latest[rid] = ((available, version), dict(row))
        result[table] = sorted(
            [r for _, r in latest.values() if r["record_status"] == "active"],
            key=lambda r: (timestamp(r["event_time"]), r["record_id"]),
        )
    if len(result["patients"]) != 1:
        raise ValueError("Expected exactly one available patient record")
    return result


def tool_view(data_dir=DEFAULT_DATA, patient_id="SYN-ONC-001", index_time=INDEX_TIME):
    tables = snapshot(data_dir, patient_id, index_time)
    return {table: json.dumps({"patient_id": patient_id, "as_of": index_time,
            "table": table, "rows": rows}, ensure_ascii=False)
            for table, rows in tables.items()}


def weight_change(vitals):
    rows = [r for r in vitals if r["measurement"] == "weight" and r["unit"] == "kg"]
    if len(rows) < 2:
        return None
    ordered = sorted(rows, key=lambda r: timestamp(r["event_time"]))
    first, last = ordered[0], ordered[-1]
    baseline, current = float(first["value_num"]), float(last["value_num"])
    if baseline <= 0 or current <= 0:
        raise ValueError("Weight must be positive")
    return {"percent_loss": round(100 * (baseline-current)/baseline, 3),
            "baseline_record": first["record_id"], "current_record": last["record_id"]}
