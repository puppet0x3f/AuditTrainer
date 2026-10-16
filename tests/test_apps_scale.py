"""CPU checks for the additional APPS oracles and exact formatting rule."""

from audittrainer.apps_scale_probes import EXTRA_INPUTS, oracle, valid_output


def test_extra_probe_oracles():
    assert len(EXTRA_INPUTS) == 8
    assert all(len(texts) == 3 for texts in EXTRA_INPUTS.values())
    for root_id, texts in EXTRA_INPUTS.items():
        for text in texts:
            expected = oracle(root_id, text)
            assert valid_output(root_id, text, expected, expected)
    assert oracle('apps_train_0517', '4 1000\n') == '12'
    assert oracle('apps_train_1533', EXTRA_INPUTS['apps_train_1533'][0]) == '1'
    assert oracle('apps_train_0998', EXTRA_INPUTS['apps_train_0998'][0]) == '11'
    expected = oracle('apps_train_2597', '2\n')
    assert expected == ' 1  1  1  1\n 2  2  2 10'
    assert not valid_output('apps_train_2597', '2\n', expected.lstrip(), expected)
