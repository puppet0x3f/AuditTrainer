from audittrainer.apps_fresh_select import _candidate, _sample, _same_input


def test_public_sample_is_removed_from_heldout_cases():
    question = '''-----Sample Input-----
2

1 2
-----Sample Output-----
3

The answer is three.'''
    record = {'id': 9000, 'difficulty': 'competition', 'url': 'https://example.test/9000',
              'question': question, 'solutions': '["print(3)"]',
              'input_output': '{"inputs": ["2\\n1 2", "1\\n4", "1\\n5"], '
                              '"outputs": ["3", "4", "5"]}'}
    assert _sample(question) == {'input': '2\n1 2\n', 'expected': '3'}
    assert _same_input('2\n\n1 2', '2\n1 2\n')
    item = _candidate(record, set(), set())
    assert item['sample_group'] == 'sample'
    assert item['removed_public_indices'] == [0]
    assert item['heldout_indices'] == [1, 2]


def test_rejects_misparsed_or_conflicting_public_answer():
    question = '-----Sample Input-----\n2\n-----Sample Output-----\n9\n'
    record = {'id': 9001, 'difficulty': 'competition', 'url': 'https://example.test/9001',
              'question': question, 'solutions': '["print(2)"]',
              'input_output': '{"inputs": ["2", "3", "4"], '
                              '"outputs": ["2", "3", "4"]}'}
    assert _candidate(record, set(), set()) is None
