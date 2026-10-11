# Step 10 API evidence (live server on 127.0.0.1:8100, 2026-10-11)

All outputs below were copied from real requests to the running orchestrator (curl + a WebSocket client).
The internal key is never printed.

## Health and authentication
```
GET /health (no key)  -> {"status":"ok","predict_model_present":false,"jobs":{},...}   (before preparing the model)
GET /banks (no key)   -> 401
GET /banks (with key) -> bank_a: Augment (CTGAN), 69,546 rows, 149 fraud, 868 validated (86.8%), fidelity 0.8041 ...
```

## Jobs
| Job | Result |
|---|---|
| `20261011-163349-selftest-e659b0` | succeeded, `{"ok": true}` |
| `20261011-163349-prepare_model-376bff` | succeeded. Federated Arm 3, seed 42, 30 rounds; threshold logit 3.75; global F1 0.7955, PR-AUC 0.7261; **matches Step 9 seed 42: true** |
| `20261011-163506-generate-1e1ee0` (replay, all banks) | succeeded. A 110.9 s, B 99.2 s, C 10.3 s, D 10.2 s; candidates + validated files **identical to the experiment files for all four banks**; LLM banks: **0 API calls** (25 cached batches each) |
| `20261011-163506-run-6921e8` (Arm 3, seed 43) | succeeded. **Identical to Step 9's seed-43 Arm 3** (global test, every round, local tests); 30 round events streamed |

## WebSocket `/ws/jobs/{id}` (prepare_model job)
`results/api_demo/ws_prepare_model_stream.jsonl`: `job_status: running`, then 30 `round` events (global PR-AUC 0.7060 → 0.7261, each listing banks A–D), then `job_status: succeeded` and `end`.

## /predict (federated Arm 3 model)
```
fraud example (true label 1)   -> score_logit 8.5424, fraud_probability 0.9998, threshold_logit 3.75, is_fraud true
genuine example (true label 0) -> score_logit -4.4133, fraud_probability 0.0120, threshold_logit 3.75, is_fraud false
malformed body (Amount -5, V-columns missing) -> HTTP 422
POST /runs {"arms":[7]} -> HTTP 422 "arms must be 'all' or a list of numbers 1-6"
```

## Export scoping (D-045)
```
Bank C's operator (X-Bank-Scope: bank_c) asks for bank_b -> HTTP 403 "only bank_b's own operator may export bank_b's synthetic data"
Bank B's operator (X-Bank-Scope: bank_b)                  -> HTTP 200, 814 rows, all Class == 1 (synthetic fraud only)
results/api_audit/exports.log:
{"user": "pradeepa", "scope": "bank_c", "bank": "bank_b", "allowed": false}
{"user": "prachi",   "scope": "bank_b", "bank": "bank_b", "allowed": true}
```
(The downloaded test file was deleted straight afterwards.)
