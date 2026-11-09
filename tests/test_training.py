import unittest
from train import batch_settings,validate_generation_groups
from execution import program_score
from evaluate import expanded_score


class TrainingTests(unittest.TestCase):
    def test_four_prompts_eight_candidates_per_update(self):
        config=batch_settings(4,8,2)
        self.assertEqual(config['per_device_train_batch_size']*config['gradient_accumulation_steps'],32)
        self.assertEqual(validate_generation_groups([i for i in range(4) for _ in range(8)],4,8)['distinct_prompts'],4)
        with self.assertRaises(ValueError):validate_generation_groups([1]*32,4,8)
        with self.assertRaises(ValueError):validate_generation_groups([1]*31,4,8)

    def test_training_and_development_use_their_declared_scorers(self):
        class FakeSandbox:
            def score(self,code,tests):
                self.tests=tests
                return {'passed':1,'total':1,'status':'ok','errors':[]}
            def score_plus(self,code,task):return {'status':'ok','correct':False,'suites':{'plus':{'status':'fail'}}}
        sandbox=FakeSandbox()
        self.assertTrue(program_score(sandbox,'code',['x=1','assert x==1'],'opencoder/1')['correct'])
        self.assertEqual(sandbox.tests,['x=1\nassert x==1'])
        self.assertFalse(expanded_score(sandbox,'code',['assert f()==1'],1)['correct'])
        with self.assertRaises(ValueError):program_score(sandbox,'code',[])


if __name__=='__main__':unittest.main()
