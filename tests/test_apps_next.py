"""Small CPU checks for the frozen APPS diagnostic controls."""

from audittrainer import apps_next


def test_probe_oracles_and_constructive_scoring():
    assert set(apps_next.PROBE_INPUTS) == {
        'apps_train_0527', 'apps_train_0676', 'apps_train_0688', 'apps_train_0836',
        'apps_train_1311', 'apps_train_1371', 'apps_train_1557', 'apps_train_2617',
    }
    assert all(len(inputs) == 3 for inputs in apps_next.PROBE_INPUTS.values())
    for root_id, inputs in apps_next.PROBE_INPUTS.items():
        for text in inputs:
            expected = apps_next.oracle(root_id, text)
            if root_id == 'apps_train_1311':
                assert expected is None
            else:
                assert apps_next.valid_output(root_id, text, expected, expected)
    assert apps_next.oracle('apps_train_0527', apps_next.PROBE_INPUTS['apps_train_0527'][0]) == 'NO\nYES\nNO\nYES\nYES'
    assert apps_next.oracle('apps_train_0676', apps_next.PROBE_INPUTS['apps_train_0676'][1]) == 'alpha'
    assert apps_next.oracle('apps_train_2617', apps_next.PROBE_INPUTS['apps_train_2617'][0]).splitlines()[0] == '[]'

    example = '2\n2 1\n3 3\n'
    assert apps_next.valid_output('apps_train_1311', example, '1 -2\n1 2 3', None)
    assert apps_next.valid_output('apps_train_1311', example, '1 -2\n1 2 3\n', None)
    assert not apps_next.valid_output('apps_train_1311', example, '1 2\n1 2 3', None)
    assert not apps_next.valid_output('apps_train_1311', example, '1 -2\n1 2', None)
    assert not apps_next.valid_output('apps_train_1311', example, '1 -2\n1 2 3\nextra', None)
