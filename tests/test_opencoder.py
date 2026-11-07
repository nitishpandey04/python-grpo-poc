import unittest
from educational_oracles import independent_cases, matching_oracle, longest_unique_length
from opencoder_data import code_key, literal_examples, normalize, split_group, assertion_suite_score,contract_group


class OpenCoderTests(unittest.TestCase):
    def test_ast_clone_detection_ignores_cosmetic_names(self):
        a='def increment(x):\n "description"\n return x+1'
        b='def add_one(value: int) -> int:\n return value+1'
        self.assertEqual(code_key(a),code_key(b))
        self.assertNotEqual(code_key(a),code_key('def decrement(x): return x-1'))

    def test_requires_direct_literal_contracts(self):
        self.assertEqual(literal_examples(['assert f(1, b=2) == 3'],'f'), [((1,),{'b':2},3)])
        for source in ['x=1\nassert f(x)==2','assert f(x)==2','assert other(1)==2','assert f(1)','assert f(1) in [2,3]']:
            with self.assertRaises((ValueError,TypeError)):literal_examples([source],'f')

    def test_reference_and_tests_stay_out_of_prompt_fields(self):
        raw={'seq_id':1,'instruction':'Return one more than the input.','entry_point':'f','code':'def f(x): return x+1',
             'testcase':['assert f(1)==2','assert f(2)==3','assert f(3)==4']}
        row=normalize(raw)
        self.assertEqual(row['task_id'],'opencoder/1')
        self.assertEqual(row['signature'],'def f(x):')
        raw['testcase']=['assert f(1)==2']*3
        with self.assertRaises(ValueError):normalize(raw)

    def test_independent_oracle_adds_edges_and_is_bounded(self):
        examples=[(('abc',),{},'cba'),(('aa',),{},'aa'),(('xy',),{},'yx')]
        name,cases=independent_cases('Reverse a string.',examples)
        self.assertEqual(name,'reverse_string')
        self.assertIn((('',),''),cases)
        self.assertIsNone(matching_oracle('Reverse a string.',[(('x'*300,),{},'x'*300)]))
        self.assertIsNone(matching_oracle('Reverse each word in a string.',examples))
        self.assertEqual(longest_unique_length('abba'),2)
        wrong=[(('abc',),{},'abc')]
        self.assertIsNone(matching_oracle('Reverse a string.',wrong))

    def test_recognized_contract_variants_share_a_group(self):
        self.assertEqual(contract_group('palindrome_exact'),contract_group('palindrome_ignore_case'))
        self.assertEqual(contract_group('is_prime'),contract_group('not_prime'))
        self.assertEqual(contract_group('sum_even_values'),contract_group('sum_odd_values'))

    def test_split_is_deterministic_and_suite_is_all_or_nothing(self):
        self.assertEqual(split_group('same'),split_group('same'))
        self.assertIn(split_group('another'),{'train','validation','confirmation'})
        class FakeSandbox:
            def score(self,code,tests):
                self.tests=tests
                return {'passed':0,'total':1,'status':'assertion_error'}
        sandbox=FakeSandbox()
        result=assertion_suite_score(sandbox,'code',['assert a','assert b'])
        self.assertEqual(sandbox.tests,['assert a\nassert b'])
        self.assertFalse(result['correct'])
        self.assertEqual(result['test_assertions'],2)


if __name__=='__main__':unittest.main()
