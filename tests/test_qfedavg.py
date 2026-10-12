"""Extension E1: q-FedAvg wiring and the maths behind our claims about it (D-059)."""
import numpy as np
import pytest
from flwr.serverapp.strategy.qfedavg import compute_delta_and_h

from ml.federated.server_app import OrderedFedAvg, OrderedQFedAvg, make_strategy


def _qffl_update(global_w, locals_, losses, q, L):
    """One q-FedAvg aggregation step exactly as Flower computes it (on copies)."""
    deltas, hs = [], []
    for lw, loss in zip(locals_, losses):
        d, h = compute_delta_and_h([global_w.copy()], [lw.copy()], q, L, loss)
        deltas.append(d[0]); hs.append(h)
    return global_w - sum(deltas) / sum(hs)


def test_default_strategy_is_fedavg(monkeypatch):
    monkeypatch.delenv("FRAUDNET_STRATEGY", raising=False)
    s, info = make_strategy()
    assert isinstance(s, OrderedFedAvg) and info == {"strategy": "fedavg"}


def test_qfedavg_reads_settings_and_global_model_loss(monkeypatch):
    monkeypatch.setenv("FRAUDNET_STRATEGY", "qfedavg")
    monkeypatch.setenv("FRAUDNET_Q", "1")
    monkeypatch.setenv("FRAUDNET_QFFL_L", "3")
    s, info = make_strategy()
    assert isinstance(s, OrderedQFedAvg) and info == {"strategy": "qfedavg", "q": 1.0, "L": 3.0}
    assert s.q == 1.0 and s.client_learning_rate == pytest.approx(1 / 3)
    assert s.train_loss_key == "global-model-loss"   # the received model's loss, not post-training loss


def test_q0_is_an_unweighted_average_not_row_weighted_fedavg():
    g = np.zeros(3)
    locals_ = [np.full(3, 1.0), np.full(3, 4.0)]
    new = _qffl_update(g, locals_, losses=[0.5, 2.0], q=0.0, L=1.0)
    assert np.allclose(new, 2.5)                     # (1 + 4) / 2, whatever the banks' sizes are


def test_positive_q_gives_the_worse_served_bank_more_say():
    g = np.zeros(3)
    locals_ = [np.full(3, 1.0), np.full(3, 4.0)]
    # Bank 2's global-model loss is higher, so with q > 0 the update moves towards bank 2 more than at q = 0.
    new_q0 = _qffl_update(g, locals_, [0.5, 2.0], q=0.0, L=1.0)
    new_q1 = _qffl_update(g, locals_, [0.5, 2.0], q=1.0, L=1.0)
    direction = np.full(3, 4.0) - np.full(3, 1.0)
    assert (new_q1 - g) @ direction / np.linalg.norm(new_q1 - g) > (new_q0 - g) @ direction / np.linalg.norm(new_q0 - g)
