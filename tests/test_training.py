import unittest
from train import batch_settings,validate_generation_groups,select_probe_rows,summarize_probe
from execution import program_score,solution_score
from evaluate import expanded_score


class TrainingTests(unittest.TestCase):
    def test_four_prompts_eight_candidates_per_update(self):
        config=batch_settings(4,8,2)
        self.assertEqual(config['per_device_train_batch_size']*config['gradient_accumulation_steps'],32)
        self.assertEqual(validate_generation_groups([i for i in range(4) for _ in range(8)],4,8)['distinct_prompts'],4)
        with self.assertRaises(ValueError):validate_generation_groups([1]*32,4,8)
        with self.assertRaises(ValueError):validate_generation_groups([1]*31,4,8)

    def test_probe_selection_is_balanced_stable_and_ignores_rewards(self):
        rows=[{'task_id':f'{label}/{i}','difficulty':label,'reward':i%2} for label in ['easy','medium'] for i in range(20)]
        selected=select_probe_rows(rows,32,42)
        self.assertEqual(sum(r['difficulty']=='easy' for r in selected),16)
        self.assertEqual([r['task_id'] for r in selected],[r['task_id'] for r in select_probe_rows(list(reversed(rows)),32,42)])
        for r in rows:r['reward']=1-r['reward']
        self.assertEqual([r['task_id'] for r in selected],[r['task_id'] for r in select_probe_rows(rows,32,42)])
        with self.assertRaises(ValueError):select_probe_rows(rows,41,42)

    def test_probe_distinguishes_mixed_zero_and_full_reward_groups(self):
        groups=[{'task_id':str(i),'difficulty':label,'rewards':rewards,'statuses':['ok' if v else 'wrong_answer' for v in rewards],'token_cap_hits':0} for i,(label,rewards) in enumerate([('easy',[0]*8),('easy',[1]*8),('medium',[1]+[0]*7)])]
        report=summarize_probe(groups,8)
        self.assertEqual(report['overall']['mixed_reward_groups'],1)
        self.assertEqual(report['overall']['all_fail_groups'],1)
        self.assertEqual(report['overall']['all_pass_groups'],1)
        self.assertEqual(report['overall']['correct_candidates'],9)
        self.assertEqual(report['by_difficulty']['medium']['solved_at_least_once'],1)
        with self.assertRaises(ValueError):summarize_probe(groups,7)
        with self.assertRaises(ValueError):summarize_probe(groups+groups[:1],8)

    def test_training_and_development_use_their_declared_scorers(self):
        class FakeSandbox:
            def score(self,code,tests):
                self.tests=tests
                return {'passed':1,'total':1,'status':'ok','errors':[]}
            def score_stdio(self,code,cases,time_limit,memory_limit_mb):
                return {'passed':1,'total':1,'status':'ok','correct':True}
            def score_plus(self,code,task):return {'status':'ok','correct':False,'suites':{'plus':{'status':'fail'}}}
        sandbox=FakeSandbox()
        self.assertTrue(solution_score(sandbox,'code',{'mode':'stdio','cases':[{'input':'1','output':'1'}],'time_limit':1,'memory_limit_mb':64})['correct'])
        self.assertTrue(program_score(sandbox,'code',['x=1','assert x==1'],'opencoder/1')['correct'])
        self.assertEqual(sandbox.tests,['x=1\nassert x==1'])
        self.assertFalse(expanded_score(sandbox,'code',['assert f()==1'],1)['correct'])
        with self.assertRaises(ValueError):program_score(sandbox,'code',[])


if __name__=='__main__':unittest.main()
