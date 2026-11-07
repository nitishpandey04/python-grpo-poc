import json
from pathlib import Path
import unittest
from execution import Sandbox
from opencoder_data import assertion_suite_score
from reviewed_contracts import cases_for,roman_encode,next_perm,family,protected_hits
from strengthen_opencoder import controls_for,make_suite

ROOT=Path(__file__).resolve().parents[1]
ROWS={r['entry_point']:r for r in map(json.loads,(ROOT/'experiments/opencoder-data-v1/mixed-train.jsonl').read_text().splitlines())}


class ReviewedOracleTests(unittest.TestCase):
    def test_known_roman_and_permutation_edges(self):
        self.assertEqual(roman_encode(1994),'MCMXCIV')
        self.assertEqual(roman_encode(3999),'MMMCMXCIX')
        self.assertEqual(next_perm([2,1,1]),[1,1,2])
        self.assertEqual(next_perm([1,1,2]),[1,2,1])
        self.assertEqual(next_perm([]),[])

    def test_specified_boundaries_are_in_generated_tests(self):
        for kind,entry,inputs,expected in [
            ('hex_color','is_valid_hex_color',('#123456\n',),False),
            ('count_digit','count_digit',(0,0),1),
            ('attendance','check_attendance',('LALL',),True),
            ('crossword','longest_row_length',(['ABC DE'],),3),
            ('permutation_1n','is_valid_permutation',([1,1,4,4],),False)]:
            pairs,_=cases_for(kind,ROWS[entry])
            actual={repr(args):value for args,value in pairs}
            self.assertEqual(actual[repr(inputs)],expected)

    def test_related_contracts_cannot_get_separate_weights(self):
        self.assertEqual(family('consecutive_dedup'),family('consecutive_duplicates'))
        self.assertEqual(protected_hits('third_largest',[{'task_id':1,'instruction':'Find the second highest number in a list.'}]),[1])
        self.assertEqual(protected_hits('power_two',[{'task_id':2,'instruction':'Find the smallest power of 2 above n.'}]),[2])


class ReviewedSuiteDockerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.sandbox=Sandbox();cls.sandbox.start()
    @classmethod
    def tearDownClass(cls):cls.sandbox.close()

    def test_stronger_tests_catch_real_source_reference_bugs(self):
        for entry,kind in [('is_valid_hex_color','hex_color'),('longest_row_length','crossword'),('count_digit','count_digit')]:
            row=ROWS[entry]
            self.assertTrue(assertion_suite_score(self.sandbox,row['reference'],row['tests'])['correct'])
            tests,_,_=make_suite(row,kind,42)
            self.assertFalse(assertion_suite_score(self.sandbox,row['reference'],tests)['correct'],entry)

    def test_good_reference_passes_and_all_faulty_controls_fail(self):
        row=ROWS['check_attendance'];tests,_,_=make_suite(row,'attendance',42)
        self.assertTrue(assertion_suite_score(self.sandbox,row['reference'],tests)['correct'])
        for control in controls_for(row,'attendance'):
            self.assertFalse(assertion_suite_score(self.sandbox,control['code'],tests)['correct'],control['name'])


if __name__=='__main__':unittest.main()
