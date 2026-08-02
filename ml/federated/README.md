# ml/federated/

Phase 4. Flower server (FedAvg strategy) and the four bank `ClientApp`s, round orchestration, and per-round metric logging. Only serialized model weight updates cross the client boundary — never raw or synthetic rows. Written against the pinned Flower version in `ml/requirements.txt` (D7).
