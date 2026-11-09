import unittest
from prepare_data import normalize,normalized,same_statement


def problem(tests,reference=None):
    row={'seq_id':1,'instruction':'Return one more than x.','entry_point':'f','testcase':tests}
    if reference is not None:row['code']=reference
    return row


class NativeDataTests(unittest.TestCase):
    def test_one_test_and_missing_or_broken_reference_are_eligible(self):
        for reference in [None,'def broken(:','def f(x): return 999']:
            row=normalize(problem(['assert f(1)==2'],reference))
            self.assertEqual(row['tests'],['assert f(1)==2'])
            self.assertNotIn('999',row['prompt'][-1]['content'])

    def test_setup_helpers_loops_and_non_equality_checks_are_preserved(self):
        chunks=['values=[1,2,3]','def check():',' for value in values:','  assert f(value)>value','check()']
        self.assertEqual(normalize(problem(chunks))['tests'],['\n'.join(chunks)])

    def test_raise_based_tests_do_not_require_assertions(self):
        tests=['if f(1)!=2:',' raise ValueError("wrong result")']
        self.assertEqual(normalize(problem(tests))['tests'],['\n'.join(tests)])

    def test_reference_classes_and_length_do_not_filter_problem(self):
        code='class Helper: pass\n'+'# comment\n'*500+'def f(x): return x+1'
        self.assertEqual(normalize(problem(['assert f(1)==2'],code))['task_id'],'opencoder/1')

    def test_syntax_errors_and_setup_without_checks_are_reported(self):
        with self.assertRaisesRegex(ValueError,'test_syntax_error'):normalize(problem(['assert (']))
        with self.assertRaisesRegex(ValueError,'no_explicit_test_check'):normalize(problem(['values=[1,2,3]']))

    def test_duplicate_matching_preserves_numbers_and_negation(self):
        self.assertTrue(same_statement(normalized('Write a function.'),normalized('Write a function.')))
        prefix='Write a Python function that examines the elements of the supplied list and returns whether '
        self.assertFalse(same_statement(normalized(prefix+'all numbers are divisible by 2.'),normalized(prefix+'all numbers are divisible by 3.')))
        self.assertFalse(same_statement(normalized(prefix+'the list is sorted.'),normalized(prefix+'the list is not sorted.')))


if __name__=='__main__':unittest.main()
