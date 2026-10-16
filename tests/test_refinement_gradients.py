"""Weighted decomposition must reproduce a real accumulated loss gradient."""
import pytest
import torch

from audittrainer.refinement_gradients import flat_gradient, geometry
from audittrainer.refinement_pr1 import conditional_token_terms, mode_surrogate


def test_weighted_components_match_accumulated_gradient_and_detect_opposition():
    parameter = torch.tensor([.2, -.1], requires_grad=True)
    unused = torch.tensor([1.], requires_grad=True)
    parameters = [parameter, unused]
    vectors = {name: torch.zeros(3) for name in ('mode', 'body', 'kl', 'sft')}
    for q, advantage in [([.8, .2], -.5), ([.1, .7], .4)]:
        current = parameter.log_softmax(0)
        old = current.detach()
        mode = -mode_surrogate(current, old, q, [q[1]]*3, 'B3', .2)/256/2
        logits = parameter.repeat(3, 1)
        reference = torch.zeros_like(logits)
        old_tokens = logits.detach().log_softmax(-1)[range(3), [0, 1, 0]]
        objective, kl, _ = conditional_token_terms(logits, reference, [0, 1, 0], old_tokens, advantage, .2)
        body = -objective*old[1].exp()/3/256/2
        penalty = .02*kl*old[1].exp()/3/256/2
        sft = .02*parameter.square().sum()/256/2
        for name, loss in zip(vectors, [mode, body, penalty, sft]):
            vectors[name] += flat_gradient(loss, parameters, retain_graph=True)
        (mode+body+penalty+sft).backward()
    expected = torch.cat([parameter.grad, torch.zeros(1)])
    torch.testing.assert_close(sum(vectors.values()), expected)
    assert unused.grad is None
    pairs = geometry({'a': torch.tensor([1., 0.]), 'b': torch.tensor([-2., 0.]), 'zero': torch.zeros(2)})
    assert pairs['pairs']['a:b']['cosine'] == pytest.approx(-1)
    assert pairs['pairs']['a:zero']['cosine'] is None
