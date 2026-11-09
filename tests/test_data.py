import unittest
from prepare_data import normalize,normalized,same_statement,exclusion,assign_split


def problem():
    return {'id':'1/A','description':'Add the given numbers.','title':'Addition','input_format':'Two integers.','output_format':'Their sum.','examples':[{'input':'1 2','output':'3'}],
        'rating':1400,'official_tests':[{'input':'2 3','output':'5'}],'testset_size':1,'time_limit':1,'memory_limit':256,'executable':True,'official_tests_complete':True,'input_mode':'stdio','interaction_format':None,'generated_checker':None}


class CodeforcesDataTests(unittest.TestCase):
    def test_medium_tasks_are_python_only_and_hidden_tests_are_not_in_prompt(self):
        row=normalize(problem())
        self.assertEqual(row['language'],'python');self.assertEqual(row['difficulty'],'medium')
        self.assertIn('Python 3',row['prompt'][0]['content'])
        self.assertNotIn('2 3',row['prompt'][-1]['content'])
        self.assertEqual(row['tests']['cases'],[{'input':'2 3','output':'5'}])

    def test_requested_rating_scope_keeps_medium_and_defers_hard(self):
        row=problem()
        for rating in [800,1000,1100,1400,1600]:
            row['rating']=rating;self.assertIsNone(exclusion(row))
        row['rating']=1700;self.assertEqual(exclusion(row),'outside_requested_ratings_or_unrated')

    def test_missing_statement_and_unavailable_grader_are_explicit(self):
        row=problem();row['description']=None
        self.assertEqual(exclusion(row),'missing_problem_statement')
        row=problem();row['generated_checker']='checker code'
        self.assertEqual(exclusion(row),'custom_checker_not_supported_in_initial_scope')
        row=problem();row['official_tests_complete']=False
        self.assertEqual(exclusion(row),'additional_generated_tests_not_loaded')

    def test_missing_checker_metadata_cannot_override_output_contract(self):
        row=problem();row['output_format']='If multiple solutions exist, print any of them.'
        self.assertEqual(exclusion(row),'output_spec_requires_unavailable_semantic_checker')
        row['output_format']='Your absolute or relative error must be below 1e-6.'
        self.assertEqual(exclusion(row),'output_spec_requires_unavailable_semantic_checker')
        row['output_format']='Print YES if any solution exists. The answer is unique.'
        self.assertIsNone(exclusion(row))

    def test_one_complete_test_is_allowed_but_bad_completeness_is_rejected(self):
        row=problem();self.assertEqual(len(normalize(row)['tests']['cases']),1)
        row['testset_size']=2
        with self.assertRaises(ValueError):normalize(row)

    def test_split_does_not_depend_on_model_outcomes(self):
        self.assertEqual(assign_split('123/A'),assign_split('123/A'))
        self.assertIn(assign_split('123/A'),{'train','validation'})

    def test_duplicate_matching_preserves_numbers_and_negation(self):
        prefix='Write a Python function that examines the elements of the supplied list and returns whether '
        self.assertFalse(same_statement(normalized(prefix+'all numbers are divisible by 2.'),normalized(prefix+'all numbers are divisible by 3.')))
        self.assertFalse(same_statement(normalized(prefix+'the list is sorted.'),normalized(prefix+'the list is not sorted.')))


if __name__=='__main__':unittest.main()
