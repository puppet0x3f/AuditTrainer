from collections import Counter
import json
from pathlib import Path
import random
from types import SimpleNamespace

import yaml

from audittrainer.dataset import load_root_manifest, read_jsonl
from audittrainer.refinement_depth2_local import INTERNAL, LocalPlanner
from audittrainer.refinement_responsibility_course import CourseReference, prepare
from audittrainer.refinement_rollout import run_case


def test_internal_seed_reset_controls_executor_after_stochastic_planning(tmp_path):
    study = yaml.safe_load(Path('configs/refinement_responsibility_course.yaml').read_text())
    study.update(output_dir=str(tmp_path/'course'), data_dir=str(tmp_path/'data'))
    config, framework, _ = prepare(study)
    row = load_root_manifest(config['root_manifest'])['train_course_calendar_matrix_04']
    tests = {r['root_id']: r for r in read_jsonl(config['test_manifest'])}
    definition = json.loads(Path(row['fixture_path']).read_text())
    reference = json.loads(Path(row['reference_path']).read_text())
    observed = []
    for reset in (False, True):
        teacher = CourseReference(definition, reference, 'recursive')
        def execute(observation):
            if observation['node_id'] == INTERNAL and observation['phase'] == 'IMPLEMENT':
                observed.append(random.random())
            return teacher.act(observation)
        executor = SimpleNamespace(cost=Counter(), llm=SimpleNamespace(identity={}), act=execute)
        base = SimpleNamespace(cost=Counter(), identity={})
        planner = LocalPlanner(base, definition, 'calendar', 'ATOMIC')
        direct = planner.act
        def stochastic(observation):
            if observation['node_id'] == INTERNAL:
                for _ in range(17): random.random()
            return direct(observation)
        planner.act = stochastic
        case = run_case(row, tests, framework, config, planner, executor, tmp_path/str(reset), 876543,
                        reset_seed_after_root_plan=True, reset_seed_after_plan_node=INTERNAL if reset else None)
        assert case['status'] == 'COMPLETE' and case['final_scores']['private']['status'] == 'PASS'
        if reset: assert case['continuation_seed_resets_after_plan'] == [{'node_id': INTERNAL, 'seed': 876543}]
    assert observed[0] != observed[1] == random.Random(876543).random()


def test_natural_delegates_only_the_internal_decision():
    sentinel = object()
    base = SimpleNamespace(cost=Counter(), identity={}, act=lambda observation: sentinel)
    planner = LocalPlanner(base, {}, 'calendar', 'NATURAL')
    assert planner.act({'node_id': INTERNAL}) is sentinel
