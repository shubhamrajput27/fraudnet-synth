# 5-Minute Live Demo Script: Multi-Laptop Federated Learning (Step 8B)

**Before the guide arrives:** all four laptops are on Shubham's hotspot, the server and Bank A are running on Shubham's laptop (setup guide, Part 4), and each teammate has run `check_my_data` once. Keep the one-laptop fallback ready: `python -m ml.federated.demo.fallback`.

| Time | Who | Does | Says (simple English) |
|---|---|---|---|
| 0:00 | **Shubham** | Points at the four laptops | "Each laptop is one bank. Sir, you asked to see the model move between banks, so this is real network traffic over a phone hotspot, not a simulation. Each bank's transactions stay on its own laptop for the whole run." |
| 0:30 | **Sathvik** (Bank D) | Runs `python -m ml.federated.demo.check_my_data --bank bank_d` | "This proves my laptop holds only Bank D: 4 files with matching checksums. Banks A, B and C are empty, and the raw Kaggle file and the global split aren't here. If anything were wrong, my client would refuse to start." |
| 1:00 | **Prachi** (Bank B) | Runs `start_client --bank bank_b --server <IP>` | "My laptop now connects to Shubham's server and waits. It will only ever send model weights." |
| 1:15 | **Pradeepa** (Bank C) | Same for `bank_c` | "Bank C is connected too." |
| 1:30 | **Shubham** | Window 3: `run_demo` | "Now I start 10 rounds. In each round, my server sends the current model, about 17 KB, to every bank." |
| 1:45 | **Prachi** | Points at her `Round N:` lines | "Here my laptop received the global model, trained it on my 41,801 rows (164 fraud, including our validated synthetic rows), and sent back only the updated weights, 17 KB. My rows never left this laptop. Bank B is data-rich, so our synthetic fraud came from CTGAN, which learned from Bank B's own fraud rows." |
| 2:30 | **Pradeepa** | Points at her lines | "Bank C is data-poor, with only 28 training frauds, so its synthetic rows came from the LLM. It was given only summary statistics, never real rows. Those files were generated in advance on Shubham's laptop and handed to me in person." |
| 3:00 | **Shubham** | Points at the server log | "Here the server lists every update that arrived: from each bank, one ArrayRecord of 6 weight tensors (17,478 bytes) plus 5 scalar metrics, and nothing else. It averages them, giving bigger banks more weight, and sends the new model back. That's FedAvg." |
| 3:45 | **Sathvik** | — | "At the end, each bank sends only counts, how many frauds and genuine transactions are above each of 241 cut-offs, to choose the alert threshold. Bank D has only 4 validation frauds, so its own numbers are noisy, which is why federation helps small banks most." |
| 4:15 | **Shubham** | Shows the final line | "Here are the global test results. This demo is to show the mechanism; the numbers in our report come from our controlled single-machine experiments." |
| 4:45 | **Shubham** | — | "To summarise: the model travelled to four laptops and back every round, while each bank's data stayed where it was." |

**Honest points to say if asked** (also in the step doc and report limitations):
- Handing out the zip files is *simulation set-up*. In real life each bank already owns its data. The privacy claim covers **training**, during which no data row leaves any laptop.
- The synthetic data was generated on Shubham's laptop (Steps 4–6), not on the teammates' laptops.
- The server laptop holds the global test set to measure results. That's a disclosed design choice.
- Code can check what's on a laptop, but it can't stop a person deliberately copying files. That's why the handover is in person and the zips are deleted.
