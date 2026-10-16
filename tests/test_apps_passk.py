"""Check the cumulative pass@k aggregate on paired candidate outcomes."""

import pytest

from audittrainer.apps_passk import cumulative_pass_counts


def test_cumulative_pass_counts():
    attempts = {'a': [True, False, False], 'b': [False, False, True],
                'c': [False, True, False]}
    assert cumulative_pass_counts(attempts) == [1, 2, 3]
    with pytest.raises(ValueError):
        cumulative_pass_counts({'a': [True], 'b': [False, True]})
