"""Federated arm (CLAUDE.md six-arm grid): FedAvg across the 4 banks, real or
real+validated-synthetic per client. Hand-rolled sequential round loop (see client.py docstring
for why — no ray/Windows/Python-3.13 wheel), but the actual weight aggregation is real Flower:
`flwr.server.strategy.FedAvg.aggregate_fit`, fed genuine `flwr.common.FitRes` objects. Verified
empirically that `aggregate_fit` never touches the `ClientProxy` object in each result tuple
(only `fit_res.parameters`/`fit_res.num_examples`), so a `None` placeholder stands in for it.

Only weight arrays (`Parameters`) ever cross the client/server boundary here — never a row of
data, real or synthetic (the privacy invariant, CLAUDE.md hard constraints).

Evaluation: after each round's aggregation, the new global model is evaluated by every client
against the shared holdout using that client's own scaler (D4; confirmed with the user 2026-08-03
— see PLAN.md), then macro-averaged. The last round's macro-average is the arm's headline number,
and every round's per-client numbers are logged for the eventual live round charts (Phase 6).
"""
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from ml.common.artifacts import save_manifest, save_model, save_scaler
from ml.common.data import BANKS, fit_scaler, load_client_training_data, load_holdout, to_arrays
from ml.common.metric_logger import RunLogger
from ml.common.metrics import macro_average
from ml.common.model import get_parameters, new_model, set_parameters
from ml.federated.client import FedClient


def _build_clients(augmented: bool, seed: int) -> list[FedClient]:
    holdout = load_holdout()
    clients = []
    for bank in BANKS:
        train_df = load_client_training_data(bank, augmented)
        scaler = fit_scaler(train_df)
        X_train, y_train = to_arrays(train_df, scaler)
        X_eval, y_eval = to_arrays(holdout, scaler)
        model = new_model(seed)
        clients.append(FedClient(bank, model, X_train, y_train, X_eval, y_eval, seed, scaler=scaler))
    return clients


def run_federated(
    augmented: bool,
    seed: int,
    num_rounds: int = 10,
    local_epochs: int = 2,
    run_id: str | None = None,
    save_artifacts: bool = False,
) -> dict:
    arm = "federated_augmented" if augmented else "federated_real"
    logger = RunLogger(
        arm=arm, seed=seed,
        config={"augmented": augmented, "num_rounds": num_rounds, "local_epochs": local_epochs},
        run_id=run_id,
    )

    clients = _build_clients(augmented, seed)
    strategy = FedAvg()
    global_params = get_parameters(new_model(seed))

    last_round_metrics: list[dict] = []
    for round_num in range(1, num_rounds + 1):
        fit_results = []
        for client in clients:
            new_params, num_examples, fit_metrics = client.fit(global_params, {"local_epochs": local_epochs})
            fit_res = FitRes(
                status=Status(Code.OK, ""),
                parameters=ndarrays_to_parameters(new_params),
                num_examples=num_examples,
                metrics=fit_metrics,
            )
            fit_results.append((None, fit_res))

        aggregated_parameters, _ = strategy.aggregate_fit(round_num, fit_results, failures=[])
        global_params = parameters_to_ndarrays(aggregated_parameters)

        round_client_metrics = []
        for client in clients:
            _, _, eval_metrics = client.evaluate(global_params, {})
            logger.log_client_metric(round_num=round_num, bank=client.bank, metrics=eval_metrics)
            round_client_metrics.append(eval_metrics)

        round_macro = macro_average(round_client_metrics)
        logger.log_round_metric(round_num=round_num, metrics=round_macro)
        last_round_metrics = round_client_metrics
        print(f"  Round {round_num}/{num_rounds}: macro f1={round_macro['f1']:.4f} "
              f"precision={round_macro['precision']:.4f} recall={round_macro['recall']:.4f}")

    final = macro_average(last_round_metrics)
    logger.finalize(final)

    if save_artifacts:
        global_model = new_model(seed)
        set_parameters(global_model, global_params)
        save_model(logger.run_id, global_model)  # one shared model — federated has no per-bank model
        for client in clients:
            save_scaler(logger.run_id, client.scaler, bank=client.bank)
        save_manifest(logger.run_id, arm, banks=BANKS)

    return {"run_id": logger.run_id, "arm": arm, "final_metrics": final, "per_client": last_round_metrics}
