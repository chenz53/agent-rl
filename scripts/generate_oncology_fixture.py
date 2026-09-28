"""Generate deterministic, wholly invented oncology EHR exports; no real data."""
import csv
import gzip
import io
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoastra.oncology import ROOT, COMMON, TABLES, INDEX_TIME
from nanoastra.oncology_outcomes import COLUMNS


def write_gzip(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    # Zero mtime and no original filename: repeated generation is byte-identical.
    with path.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as zipped:
            zipped.write(text.getvalue().encode("utf-8"))


def generate(destination=ROOT / "fixtures/oncology"):
    destination = Path(destination)
    rows = {table: [] for table in TABLES}
    p = "SYN-ONC-001"
    def add(table, rid, event, available=None, patient=p, version=1, status="active", encounter="ENC-001", **values):
        row = dict.fromkeys(COMMON + TABLES[table], "")
        row.update(record_id=rid, version=str(version), patient_id=patient,
                   encounter_id=encounter, event_time=event, available_time=available or event,
                   record_status=status, source_system="SYNTHETIC_EHR")
        row.update({k: str(v) for k,v in values.items()})
        rows[table].append(row)
    add("patients", "PAT-001", "2025-01-05T10:00:00Z", age_at_index=67, sex="female", birth_year=1957, encounter="")
    add("encounters", "ENCROW-001", "2025-02-28T14:00:00Z", encounter_type="outpatient", department="medical_oncology", reason="Initial systemic therapy planning")
    add("diagnosis", "DX-001", "2025-02-10T10:00:00Z", code_system="LOCAL", code="LUNG_ADENO", description="Lung adenocarcinoma", verification_status="pathology_confirmed")
    add("diagnosis", "DX-002", "2025-02-18T11:00:00Z", code_system="LOCAL", code="MET_BONE_LIVER", description="Metastatic disease involving liver and bone", verification_status="clinical_imaging")
    add("diagnosis", "DX-003", "2024-11-10T10:00:00Z", code_system="LOCAL", code="COPD", description="Chronic obstructive pulmonary disease", verification_status="historical_problem_list", encounter="ENC-HIST")
    add("pathology", "PATH-001", "2025-02-07T11:00:00Z", "2025-02-10T10:00:00Z", specimen_site="right lung mass", histology="adenocarcinoma", report_text="Biopsy: adenocarcinoma compatible with lung primary; TTF-1 positive. Limited tissue remains for molecular analysis.")
    add("staging", "STG-001", "2025-02-18T11:00:00Z", stage_system="clinician_documented_group", stage_group="IV", basis="clinical", disease_sites="lung; liver; bone")
    add("procedures", "PROC-001", "2025-02-18T09:00:00Z", "2025-02-18T11:00:00Z", procedure_name="CT chest abdomen pelvis", procedure_status="completed", result_summary="Right lung mass, multiple liver lesions and lytic bone lesions; interpreted by oncology as metastatic disease.")
    add("procedures", "PROC-002", "2025-02-22T10:00:00Z", procedure_name="Brain MRI", procedure_status="ordered_not_performed", result_summary="No imaging report available")
    for rid,date,weight in [("WT-001","2025-01-05T10:00:00Z",72),("WT-002","2025-02-12T10:00:00Z",68),("WT-003","2025-02-28T14:00:00Z",64)]:
        add("vitals",rid,date,measurement="weight",value_num=weight,unit="kg",context="outpatient scale")
    add("vitals","HR-001","2025-02-28T14:00:00Z",measurement="heart_rate",value_num=96,unit="beats/min",context="rest")
    add("vitals","O2-001","2025-02-28T14:00:00Z",measurement="spo2",value_num=94,unit="%",context="room air")
    labs=[("HB-001","2025-02-12T10:00:00Z","hemoglobin",11.2,"g/dL",12,16),
          ("HB-002","2025-02-28T08:00:00Z","hemoglobin",10.1,"g/dL",12,16),
          ("ALB-001","2025-02-12T10:00:00Z","albumin",35,"g/L",35,50),
          ("ALB-002","2025-02-28T08:00:00Z","albumin",4.9,"g/dL",3.5,5.0),
          ("CR-001","2025-02-28T08:00:00Z","creatinine",0.9,"mg/dL",0.5,1.1),
          ("NEUT-001","2025-02-28T08:00:00Z","neutrophils",8.0,"10^9/L",1.8,7.5),
          ("LYM-001","2025-02-28T08:00:00Z","lymphocytes",1.0,"10^9/L",1.0,4.0)]
    for rid,date,name,val,unit,lo,hi in labs:
        add("labs",rid,date,test_code="LOCAL_"+name.upper(),test_name=name,value_num=val,unit=unit,ref_low=lo,ref_high=hi,specimen_id="SP-"+date[:10])
    add("labs","ALB-002","2025-02-28T08:00:00Z","2025-02-28T16:00:00Z",version=2,test_code="LOCAL_ALBUMIN",test_name="albumin",value_num=3.1,unit="g/dL",ref_low=3.5,ref_high=5.0,specimen_id="SP-2025-02-28",value_text="Corrected result")
    add("labs","LDH-001","2025-02-28T08:00:00Z","2025-03-01T11:00:00Z",test_code="LOCAL_LDH",test_name="LDH",value_num=480,unit="U/L",ref_low=120,ref_high=250,specimen_id="SP-2025-02-28")
    add("performance_status","PS-001","2025-02-12T10:00:00Z",scale="ECOG",score=1,assessor="oncologist",context="ambulatory")
    add("performance_status","PS-002","2025-02-28T14:00:00Z",scale="ECOG",score=2,assessor="oncologist",context="Ambulatory and capable of self-care; unable to work; up more than half of waking hours")
    add("biomarkers","BIO-001","2025-02-07T11:00:00Z","2025-02-15T12:00:00Z",marker="PD-L1",assay="IHC TPS",result="60%",result_status="final")
    add("biomarkers","BIO-002","2025-02-20T12:00:00Z",marker="tumor_NGS_panel",assay="DNA/RNA panel",result="",result_status="pending")
    add("biomarkers","BIO-002","2025-02-20T12:00:00Z","2025-03-04T12:00:00Z",version=2,marker="tumor_NGS_panel",assay="DNA/RNA panel",result="No actionable alteration identified in tested panel",result_status="final")
    add("notes","NOTE-001","2025-02-28T14:00:00Z","2025-02-28T17:00:00Z",note_type="oncology_consult",author_role="oncologist",text="New metastatic lung adenocarcinoma. Fatigue, reduced appetite, approximately 8 kg weight loss since January. ECOG 2 today, previously 1. Lives with partner; daughter helps with shopping. Independent in basic self-care. Bone pain treated with as-needed analgesia. Systemic treatment not started; final regimen awaits molecular results and shared decision-making.\nBrain MRI remains pending. Discuss prognosis with uncertainty; no individualized survival estimate documented.")
    add("notes","NOTE-002","2025-02-28T15:00:00Z","2025-03-02T10:00:00Z",note_type="late_signed_addendum",author_role="oncologist",text="Late-entered addendum containing information not available at the index time; must not appear in the baseline view.")
    add("notes","NOTE-003","2025-04-15T10:00:00Z",note_type="followup",author_role="oncologist",text="Post-index clinical course and treatment response; excluded from baseline prognosis features.")
    add("medication_orders","ORD-001","2025-02-28T15:00:00Z",drug="systemic anticancer regimen",intent="palliative_systemic",order_status="draft_awaiting_molecular_results")
    add("medication_orders","ORD-002","2025-02-28T15:00:00Z",drug="analgesic",intent="symptom_control",order_status="active")
    add("medication_administrations","ADM-001","2025-03-06T10:00:00Z",drug="synthetic systemic regimen",dose="",unit="",route="IV",administration_status="administered")
    # Two sparse additional patients demonstrate patient isolation and censoring,
    # not a representative training cohort or comparative clinical phenotypes.
    for suffix in ["002","003"]:
        pid="SYN-ONC-"+suffix
        add("patients","PAT-"+suffix,"2025-02-01T10:00:00Z",patient=pid,age_at_index=65,sex="male",birth_year=1959,encounter="")
        add("diagnosis","DX-"+suffix+"-OTHER","2025-02-01T10:00:00Z",patient=pid,code_system="LOCAL",code="LUNG_ADENO",description="Synthetic lung adenocarcinoma",verification_status="pathology_confirmed")
    for table in TABLES:
        write_gzip(destination/"clinical"/f"{table}.csv.gz", COMMON+TABLES[table], rows[table])
    outcomes=[
        dict(zip(COLUMNS,[p,"2025-06-09T09:00:00Z","2025-06-08T09:00:00Z","2025-12-31T09:00:00Z"])),
        dict(zip(COLUMNS,["SYN-ONC-002","","2025-10-01T09:00:00Z","2025-12-31T09:00:00Z"])),
        dict(zip(COLUMNS,["SYN-ONC-003","","2025-04-30T09:00:00Z","2025-12-31T09:00:00Z"])),
    ]
    write_gzip(destination/"evaluator_only"/"mortality.csv.gz",COLUMNS,outcomes)
    manifest={"synthetic":True,"index_time":INDEX_TIME,"horizon_days":180,
              "task":"Pretreatment all-cause mortality prognosis; teaching fixture only",
              "clinical_tables":list(TABLES),"row_counts_including_versions":{t:len(r) for t,r in rows.items()},
              "patients":3,"source":"Entirely invented; not sampled from a real clinical system"}
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))

if __name__ == "__main__":
    generate()
