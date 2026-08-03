"""ml/common/model.py: D2 architecture + state_dict<->ndarray round-trip (the exact form Flower's
NumPyClient/FedAvg exchange — Phase 4's federated loop depends entirely on this being exact)."""
import numpy as np
import torch

from ml.common.model import N_FEATURES, get_parameters, new_model, set_parameters


def test_new_model_is_seeded_deterministically():
    m1 = new_model(seed=42)
    m2 = new_model(seed=42)
    for p1, p2 in zip(get_parameters(m1), get_parameters(m2)):
        assert np.array_equal(p1, p2)


def test_different_seeds_give_different_weights():
    m1 = new_model(seed=42)
    m2 = new_model(seed=1)
    params1, params2 = get_parameters(m1), get_parameters(m2)
    assert not all(np.array_equal(p1, p2) for p1, p2 in zip(params1, params2))


def test_get_set_parameters_round_trip_is_exact():
    model = new_model(seed=42)
    original = [p.copy() for p in get_parameters(model)]

    # Mutate the model, then restore via set_parameters — must recover the exact original values.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    set_parameters(model, original)

    for restored, expected in zip(get_parameters(model), original):
        assert np.array_equal(restored, expected)


def test_forward_pass_shape_and_dtype():
    model = new_model(seed=42)
    x = torch.randn(5, N_FEATURES)
    logits = model(x)
    assert logits.shape == (5,)  # raw per-sample logit, squeezed — not (5, 1)
    assert logits.dtype == torch.float32
