"""Evaluator-only demonstration of event, non-event, and censored outcomes."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nanoastra.oncology import ROOT, INDEX_TIME
from nanoastra.oncology_outcomes import load_labels
if __name__ == '__main__':
    print(json.dumps(load_labels(ROOT / 'fixtures/oncology/evaluator_only/mortality.csv.gz', INDEX_TIME), indent=2))
