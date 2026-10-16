import json
from pathlib import Path

import pytest
import yaml

from audittrainer.refinement_body_fit import prepare


def study_at(tmp_path):
    study = yaml.safe_load(Path('configs/refinement_body_fit.yaml').read_text())
    study['output_dir'] = str(tmp_path/'fit')
    scoring = json.loads(Path(study['scoring_result']).read_text())
    path = tmp_path/'scoring.json'
    path.write_text(json.dumps(scoring)); study['scoring_result'] = str(path)
    return study, scoring, path


def test_body_diagnostic_uses_corrected_tests_and_original_public_target(tmp_path):
    study, scoring, _ = study_at(tmp_path)
    config, _, target = prepare(study)
    assert config['test_manifest'] == scoring['config']['test_manifest']
    assert target == json.loads(Path(study['target']).read_text())
    assert study['optimizer_steps'] == 80 and study['conditional_split_samples'] == 64
    request = json.loads((Path(study['output_dir'])/'request.json').read_text())
    assert request['corrected_calibration']['fixed/parent_aggregate'] == 1.


@pytest.mark.parametrize('failure', ['scoring_error', 'target_no_longer_better'])
def test_body_fit_requires_reliable_target(tmp_path, failure):
    study, scoring, path = study_at(tmp_path)
    if failure == 'scoring_error': scoring['status'] = 'ERROR'
    else: scoring['groups']['fixed/parent_aggregate']['new_successes'] = 0
    path.write_text(json.dumps(scoring))
    with pytest.raises(ValueError): prepare(study)
