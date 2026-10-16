from audittrainer.apps_passk_select import choose, public_example


def test_extracts_public_example_from_common_formats():
    statement = '''-----Input-----
The first line has T.
-----Output-----
Print the answer.
-----Example-----
Input
2
1 2
3 4
Output
3
7
-----Explanation-----
This is explanatory text.
Implement solve(text: str) -> str: ...'''
    assert public_example(statement) == {'input': '2\n1 2\n3 4\n', 'expected': '3\n7'}
    statement = '''-----Sample Input:-----
1
ABC
-----Sample Output:-----
3
-----Note-----
Extra text.'''
    assert public_example(statement) == {'input': '1\nABC\n', 'expected': '3'}
    statement = '''-----Sample input-----
3 176
-----Sample output-----
6
Note: Do not print anything else.'''
    assert public_example(statement) == {'input': '3 176\n', 'expected': '6'}
    statement = '''-----Sample input-----
3
-----Sample output-----
4

There are four ways to do this.'''
    assert public_example(statement, stop_at_blank_output=True) == {'input': '3\n', 'expected': '4'}
    assert public_example('-----Input Format-----\nAn integer.\n-----Output Format-----\nA number.') is None


def test_selects_sample_pass_then_earlier_candidate():
    scores = [
        {'sample_pass': False, 'sample_status': 'RETURNED', 'has_program': True},
        {'sample_pass': True, 'sample_status': 'RETURNED', 'has_program': True},
        {'sample_pass': True, 'sample_status': 'RETURNED', 'has_program': True},
    ]
    assert [choose(scores, k, True) for k in (1, 2, 3)] == [0, 1, 1]
    assert [choose(scores, k, False) for k in (1, 2, 3)] == [0, 0, 0]
