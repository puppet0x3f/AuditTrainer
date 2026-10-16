"""Focused CPU checks for APPS independent-input reference answers."""

from audittrainer.apps_scale_followup import PROBE_INPUTS, oracle


def test_followup_oracles():
    assert len(PROBE_INPUTS) == 5
    assert all(len(texts) == 3 and len(set(texts)) == 3 for texts in PROBE_INPUTS.values())
    assert oracle('apps_train_0573', '3\n1 5\n2 5\n5 3\n') == '0\n5\n8'
    assert oracle('apps_train_0971', '2\n5\n1 2 2 2 3\n4\n1 2 3 4\n') == '2\n3'
    assert oracle('apps_train_1093', '3\n2 3 5\n3\n1 3 7\n2 2 1\n3 3 11\n') == '2\n0\n5'
    assert oracle('apps_train_1510', '2\nABCDEFGHIJ\nAAABCDEFGHIJ\n') == '10\n12'
    assert oracle('apps_train_2363', '2\n4\n7 9 3 1\n4\n2 2 2 10\n') == '2\n0'
