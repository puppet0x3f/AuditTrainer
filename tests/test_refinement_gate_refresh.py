"""The scalar gate target must solve the declared regularized-value problem."""
import math
import pytest
from audittrainer.refinement_gate_refresh import target_for

@pytest.mark.parametrize('delta', [-0.75, -0.25, 0.25, 0.75])
def test_target_is_stationary_regularized_value_maximizer(delta):
    prior=0.22;beta=0.1
    p=target_for({'delta':delta,'body_prior_p_split':prior,'initial_p_split':1e-8},beta,0.125)
    derivative=delta-beta*(math.log(p/(1-p))-math.log(prior/(1-prior)))
    assert abs(derivative)<1e-10
    assert -beta*(1/p+1/(1-p))<0
    assert (p>prior)==(delta>0)

@pytest.mark.parametrize('delta', [-0.125, 0.0, 0.125])
def test_uncertain_values_preserve_initial_gate_target(delta):
    p=target_for({'delta':delta,'body_prior_p_split':0.22,'initial_p_split':0.000001},0.1,0.125)
    assert p==0.000001
