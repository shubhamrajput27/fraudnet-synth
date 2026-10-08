"""Privacy invariant (CLAUDE.md 2.4): what does a Flower client send to the server?

We call the real client handlers with a real model and inspect EVERY reply:
  * train    -> exactly the model's weight arrays (same names/shapes as the model) + scalar metrics
  * evaluate -> scalar metrics only
  * query    -> integer counts / scalar metrics only (threshold grid counts, local-test metrics)
No reply may contain a ConfigRecord, a data-shaped array, or any per-row values.
Uses Bank D's real local files (smallest bank); skipped if the data hasn't been prepared.
"""
import numpy as np
import pytest
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MessageType, RecordDict

from ml.data.load import PROJECT_ROOT, load_config
from ml.federated.client_app import evaluate_fn, query_fn, train_fn
from ml.federated.common import BANKS, threshold_grid
from ml.models.classifier import build_model
from ml.models.train import set_seed

pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / "data/clients/bank_d/train.csv").exists(),
                                reason="run `python -m ml.data.prepare` first")

BANK_IDX = BANKS.index("bank_d")


@pytest.fixture(autouse=True)
def _flower_task_identity():
    """flwr 1.39 needs a run/node/task identity to build Messages outside a running app."""
    from flwr.supercore.task_identity import TaskIdentity
    saved = (TaskIdentity._run_id, TaskIdentity._node_id, TaskIdentity._task_id)
    TaskIdentity.run_id, TaskIdentity.node_id, TaskIdentity.task_id = 1, 1, 1
    yield
    TaskIdentity._run_id, TaskIdentity._node_id, TaskIdentity._task_id = saved


def _ctx():
    return Context(run_id=1, node_id=1, node_config={"partition-id": BANK_IDX}, state=RecordDict(), run_config={})


def _msg(kind, extra=None, augmented=True):
    set_seed(0)
    model = build_model(load_config("model"))
    cfg = {"augmented": augmented, "seed": 42, "server-round": 1, **(extra or {})}
    content = RecordDict({"arrays": ArrayRecord(model.state_dict()), "config": ConfigRecord(cfg)})
    return Message(content=content, dst_node_id=1, message_type=kind), model


def _assert_only_scalars_or_count_lists(metric_record, max_list_len):
    for k in metric_record.keys():
        v = metric_record[k]
        if isinstance(v, list):
            assert len(v) <= max_list_len, f"{k}: list of {len(v)} values looks row-shaped"
            assert all(isinstance(i, int) for i in v), f"{k}: only integer counts may be sent as lists"
        else:
            assert isinstance(v, (int, float)), f"{k}: non-scalar metric"


def test_train_reply_is_weights_plus_scalars():
    msg, model = _msg(MessageType.TRAIN)
    reply = train_fn(msg, _ctx())
    c = reply.content
    assert len(c.config_records) == 0, "clients must not send config/data records"
    assert len(c.array_records) == 1 and len(c.metric_records) == 1
    arrays = next(iter(c.array_records.values()))
    # Exactly the model's parameters: same tensor names and shapes, nothing extra.
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    got = {k: tuple(v.shape) for k, v in arrays.items()}
    assert got == expected
    assert sum(np.prod(s) for s in got.values()) == 4161
    _assert_only_scalars_or_count_lists(next(iter(c.metric_records.values())), max_list_len=0)
    # The upload is the size of 4,161 float32 weights, not of any dataset.
    assert arrays.count_bytes() < 40_000


def test_evaluate_reply_is_scalars_only():
    msg, _ = _msg(MessageType.EVALUATE)
    c = evaluate_fn(msg, _ctx()).content
    assert len(c.array_records) == 0 and len(c.config_records) == 0 and len(c.metric_records) == 1
    _assert_only_scalars_or_count_lists(next(iter(c.metric_records.values())), max_list_len=0)


def test_threshold_query_reply_is_counts_only():
    grid = threshold_grid()
    msg, _ = _msg(MessageType.QUERY, {"task": "threshold-counts", "grid": grid.tolist()})
    c = query_fn(msg, _ctx()).content
    assert len(c.array_records) == 0 and len(c.config_records) == 0
    m = next(iter(c.metric_records.values()))
    _assert_only_scalars_or_count_lists(m, max_list_len=len(grid))
    # Counts are one per GRID point (241), unrelated to the number of validation rows (4,465).
    assert len(m["tp"]) == len(m["fp"]) == len(grid)
    assert m["tp"][0] <= m["n-fraud"] and m["fp"][0] <= m["n-genuine"]
    assert all(a >= b for a, b in zip(m["tp"], m["tp"][1:]))  # counts fall as the cut-off rises


def test_local_test_query_reply_is_scalars_only():
    msg, _ = _msg(MessageType.QUERY, {"task": "local-test", "threshold": 0.0})
    c = query_fn(msg, _ctx()).content
    assert len(c.array_records) == 0 and len(c.config_records) == 0
    _assert_only_scalars_or_count_lists(next(iter(c.metric_records.values())), max_list_len=0)


def test_client_reads_only_its_own_bank_folder(monkeypatch):
    """Record every CSV path the client opens while training: all must be inside its own bank folder."""
    import pandas as pd

    from ml.federated import common
    common.bank_data.cache_clear()
    opened = []
    real_read = pd.read_csv

    def spy(path, *a, **k):
        opened.append(str(path))
        return real_read(path, *a, **k)

    monkeypatch.setattr(pd, "read_csv", spy)
    msg, _ = _msg(MessageType.TRAIN, augmented=True)
    train_fn(msg, _ctx())
    common.bank_data.cache_clear()
    assert opened, "expected the client to load its data"
    own = str(PROJECT_ROOT / "data" / "clients" / "bank_d")
    assert all(p.startswith(own) for p in opened), f"client opened files outside its bank: {opened}"
