from collections import Counter

import pytest

from audittrainer.refinement_course_train import schedule


def test_course_schedule_covers_roots_and_internal_states_with_bounded_replay():
    course = [{'root_id': str(root), 'split': 'TRAIN', 'node': node}
              for root in range(24) for node in range(1+root % 3)]
    replay = [{'split': 'TRAIN'} for _ in range(108)]
    order = schedule(course, replay, 320, 42)
    assert Counter(source for source, _ in order) == {'course': 256, 'replay': 64}
    coverage = Counter(course[i]['root_id'] for source, i in order if source == 'course')
    assert len(coverage) == 24 and max(coverage.values())-min(coverage.values()) == 1
    assert {i for source, i in order if source == 'course'} == set(range(len(course)))
    assert order == schedule(course, replay, 320, 42)
    course[0]['split'] = 'VALIDATION'
    with pytest.raises(ValueError, match='Non-TRAIN'): schedule(course, replay, 320, 42)
