import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from nanoastra.oncology import DEFAULT_DATA, INDEX_TIME, ROOT, snapshot, tool_view, weight_change
from nanoastra.oncology_outcomes import horizon_label, load_labels

class OncologyTests(unittest.TestCase):
    def test_future_and_late_records_are_absent(self):
        view=snapshot()
        ids={r['record_id'] for rows in view.values() for r in rows}
        self.assertNotIn('LDH-001',ids)
        self.assertNotIn('NOTE-002',ids)
        self.assertNotIn('NOTE-003',ids)
        self.assertEqual(view['medication_administrations'],[])
        self.assertEqual(view['biomarkers'][-1]['result_status'],'pending')

    def test_latest_available_correction_and_units(self):
        view=snapshot()
        albumin=[r for r in view['labs'] if r['test_name']=='albumin']
        self.assertEqual([(r['value_num'],r['unit']) for r in albumin],[('35','g/L'),('3.1','g/dL')])
        self.assertAlmostEqual(weight_change(view['vitals'])['percent_loss'],11.111,places=3)

    def test_patient_and_outcome_isolation(self):
        view=tool_view()
        self.assertNotIn('mortality',view)
        self.assertNotIn('death_time',json.dumps(view))
        self.assertNotIn('SYN-ONC-002',json.dumps(view))
        self.assertEqual(snapshot(patient_id='SYN-ONC-002')['labs'],[])
        with self.assertRaises(ValueError): snapshot(patient_id='UNKNOWN')

    def test_csv_multiline_notes(self):
        note=snapshot()['notes'][0]['text']
        self.assertIn('\nBrain MRI',note)
        self.assertIn('Fatigue, reduced appetite',note)

    def test_event_non_event_censored(self):
        labels=load_labels(ROOT/'fixtures/oncology/evaluator_only/mortality.csv.gz',INDEX_TIME)
        self.assertEqual([r['death_by_horizon'] for r in labels],[1,0,None])
        self.assertEqual([r['followup_days'] for r in labels],[100,214,60])

    def test_event_at_horizon_and_death_after_horizon(self):
        row={'patient_id':'X','death_time':'2025-08-28T09:00:00Z', 'last_known_alive_time':INDEX_TIME,'ascertainment_end_time':'2025-12-31T09:00:00Z'}
        self.assertEqual(horizon_label(row,INDEX_TIME)['death_by_horizon'],1)
        row['death_time']='2025-08-29T09:00:00Z'
        self.assertEqual(horizon_label(row,INDEX_TIME)['death_by_horizon'],0)
        row['death_time']='2025-01-01T09:00:00Z'
        with self.assertRaises(ValueError): horizon_label(row,INDEX_TIME)

    def test_fail_closed_and_cancellation(self):
        import shutil
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            for f in DEFAULT_DATA.glob('*.csv.gz'): shutil.copyfile(f,path/f.name)
            labpath=path/'labs.csv.gz'
            with gzip.open(labpath,'rt',newline='') as f:
                reader=csv.DictReader(f); columns=reader.fieldnames; rows=list(reader)
            def save():
                with gzip.open(labpath,'wt',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
            rows[0]['available_time']=''
            save()
            with self.assertRaises(ValueError): snapshot(path)
            rows[0]['available_time']=rows[0]['event_time']
            cancelled=dict(rows[0],version='2',record_status='cancelled',available_time='2025-02-28T17:00:00Z')
            rows.append(cancelled);save()
            self.assertNotIn(rows[0]['record_id'],[r['record_id'] for r in snapshot(path)['labs']])
            rows.append(dict(cancelled));save()
            with self.assertRaises(ValueError): snapshot(path)

if __name__ == '__main__': unittest.main()
