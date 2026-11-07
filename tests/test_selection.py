import unittest
from task_selection import assert_disjoint,calibration_bucket,validate_generation_groups,expanded_score


class SelectionTests(unittest.TestCase):
    def test_heldout_overlap_is_rejected(self):
        with self.assertRaises(ValueError): assert_disjoint([{"task_id":1}],[{"task_id":1}])
        assert_disjoint([{"task_id":1}],[{"task_id":2}],[{"task_id":3}])

    def test_mixed_task_rule_excludes_all_pass_and_all_fail(self):
        self.assertEqual(calibration_bucket(0,8),'hard')
        self.assertEqual(calibration_bucket(8,8),'easy')
        self.assertEqual(calibration_bucket(1,8),'mixed')
        self.assertEqual(calibration_bucket(7,8),'mixed')
        with self.assertRaises(ValueError): calibration_bucket(9,8)

    def test_four_distinct_groups_of_eight(self):
        ids=[i for i in range(4) for _ in range(8)]
        self.assertEqual(validate_generation_groups(ids,4,8)['candidates'],32)
        with self.assertRaises(ValueError): validate_generation_groups([1]*32,4,8)
        with self.assertRaises(ValueError): validate_generation_groups(ids[:-1],4,8)
        ids[7]=2
        with self.assertRaises(ValueError): validate_generation_groups(ids,4,8)

    def test_stronger_failure_is_not_rewarded(self):
        class FakeSandbox:
            def score(self,*args): return {'passed':3,'total':3,'status':'ok','errors':[]}
            def score_plus(self,*args): return {'status':'ok','correct':False,'suites':{'base':{'status':'pass'},'plus':{'status':'fail'}}}
        score=expanded_score(FakeSandbox(),'code',['test'],1)
        self.assertFalse(score['correct'])
        self.assertEqual(score['status'],'expanded_test_failure')


if __name__=='__main__': unittest.main()
