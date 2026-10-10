# Multi-Laptop Federated Demo: Setup Guide (Step 8B)

**For:** Prachi (Bank B), Pradeepa (Bank C), Sathvik (Bank D), and Shubham (server + Bank A).
**What we're showing:** the fraud model travels between our laptops over a phone hotspot, while each bank's data never leaves its own laptop. Only model weights (about 17 KB per round) travel.

> This is a **demo**. Our reported results come from the single-machine experiments, not from this demo.

---

## Part 1: One-time setup (every teammate, about 20 minutes, at home on normal Wi-Fi)

1. **Install** Python 3.12 (python.org, tick *"Add python.exe to PATH"*) and Git (git-scm.com).
2. **Accept** Shubham's GitHub invitation to the **private** repository. It contains code only, never data.
3. Open **PowerShell** and run:
   ```powershell
   cd $HOME\Desktop
   git clone https://github.com/shubhamrajput27/fraudnet-synth.git
   cd fraudnet-synth
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt        # downloads ~1 GB (includes CPU-only PyTorch); needs internet once
   ```
4. **Check it works:** `python -c "import flwr, torch; print('ok')"` should print `ok`.

---

## Part 2: Receive your bank file (in person, pen drive)

Shubham brings a pen drive with **only your zip** (`bank_b.zip`, `bank_c.zip` or `bank_d.zip`). Do this **at your own laptop, one person at a time**:

```powershell
cd $HOME\Desktop\fraudnet-synth
.venv\Scripts\activate
Copy-Item E:\bank_b.zip .                                    # E: = the pen drive letter; use YOUR bank
Expand-Archive .\bank_b.zip -DestinationPath data\clients\
Remove-Item .\bank_b.zip                                     # delete the zip after unzipping
python -m ml.federated.demo.check_my_data --bank bank_b
```

You should see something like:
```
  bank_b: 4 files + manifest, 59,599 real rows (116 fraud), checksums OK
  bank_a: empty | bank_c: empty | bank_d: empty
  raw Kaggle file: absent | global train/val: absent | global test: absent
RESULT: OK - this laptop holds only bank_b
```
After everyone has copied their file, Shubham **wipes the pen drive**. **Never** upload your bank folder to GitHub, WhatsApp or e-mail. (Git already ignores it, so `git status` won't show it.)

If the data is ever regenerated, you'll get a versioned file (`bank_b_v2.zip`). Delete the old folder `data\clients\bank_b\` first, then repeat the steps above.

---

## Part 3: On demo day (every teammate)

1. **Join Shubham's phone hotspot.** Not college Wi-Fi, which usually blocks laptop-to-laptop traffic. If Windows asks "Allow your PC to be discoverable?", answer **Yes** (that sets the network to *Private*).
2. **Ask Shubham for the server IP**, e.g. `192.168.43.10`.
3. **Test that you can reach the server:**
   ```powershell
   Test-NetConnection 192.168.43.10 -Port 9092
   ```
   `TcpTestSucceeded : True` means you're ready. If it says `False`, see Troubleshooting.
4. **Start your bank** (use your own bank and the real IP):
   ```powershell
   cd $HOME\Desktop\fraudnet-synth
   .venv\Scripts\activate
   python -m ml.federated.demo.start_client --bank bank_b --server 192.168.43.10
   ```
   It first runs the data check and **refuses to start** if anything is wrong. Once training starts, you'll see one line per round:
   ```
   Round 3: received global model (17.1 KB) -> trained on 41,801 rows (164 fraud) -> sent weights back (17.1 KB) | local loss 0.2447
   ```
   The same lines are saved in `results\demo_8b\client_bank_b.log`. Leave the window open until Shubham says the run is finished, then press **Ctrl+C**.

---

## Part 4: Server laptop (Shubham only)

**Once, before the demo:**
```powershell
# 1. Build the CLEAN demo folder (holds only Bank A + the global test set), from the main folder:
cd "C:\Users\shubh\Desktop\Shubham Major Project\fraudnet-synth"
.venv\Scripts\python -m ml.federated.demo.export_bank --all          # makes exports\bank_a..d.zip (checked)
.venv\Scripts\python -m ml.federated.demo.make_demo_folder --dest C:\Projects\fraudnet-demo

# 2. Allow banks to reach port 9092 on ALL network profiles (run PowerShell as Administrator):
New-NetFirewallRule -DisplayName "FraudNet Flower 9092" -Direction Inbound -Protocol TCP -LocalPort 9092 -Action Allow -Profile Any
```
Only the **server** laptop needs this rule: banks only make *outgoing* connections. Windows marks a new hotspot as *Public*, which is why the rule uses `-Profile Any`. If Windows Defender pops up "Allow access?" for Python, tick **both Private and Public**.

**On demo day** (close the main working folder; work only in `C:\Projects\fraudnet-demo`). Use **three** PowerShell windows, with `$PY = "C:\Users\shubh\Desktop\Shubham Major Project\fraudnet-synth\.venv\Scripts\python.exe"`:
```powershell
cd C:\Projects\fraudnet-demo
ipconfig                                                        # note the hotspot IPv4 address -> tell the team
& $PY -m ml.federated.demo.check_my_data --bank bank_a --server-laptop
```
| Window | Command | Leave running? |
|---|---|---|
| 1 | `& $PY -m ml.federated.demo.start_server` | yes |
| 2 | `& $PY -m ml.federated.demo.start_client --bank bank_a --server 127.0.0.1 --server-laptop` | yes |
| 3 | when all banks show "Connecting…": `& $PY -m ml.federated.demo.run_demo` (two-laptop test: add `--expected-banks 2`) | finishes by itself |

Window 3 streams the server log (which bank's update arrived, its size and record types) and saves it in `results\demo_8b\`. Bank A and the server are **two separate processes**; the server process only reads the global test set.

---

## Part 5: One-laptop fallback (if the network fails)

From the **main** working folder, a single command runs the server and all four banks as five processes, each with its own isolated folder:
```powershell
.venv\Scripts\python -m ml.federated.demo.fallback
```
Logs are saved in `results\demo_8b\fallback_<time>\`. It's the same code as the multi-laptop demo; only the IP is `127.0.0.1`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Test-NetConnection … TcpTestSucceeded : False` | Server firewall, or the phone blocks device-to-device traffic ("AP/client isolation") | Check the firewall rule on the server; try another phone's hotspot; make sure the server window is running |
| Client: `REFUSING TO START … no manifest.json` | Zip not unzipped into `data\clients\` | Re-run the `Expand-Archive` command exactly as shown |
| Client: `… does not match its manifest` | File edited or corrupted | Ask Shubham for a fresh zip; never open the CSVs in Excel and save them |
| Client: `bank_x: N data file(s) present` | Another bank's files are on this laptop | Delete them; each laptop must hold only its own bank |
| Run waits forever at "Sampled 0 nodes" | Fewer banks connected than `--expected-banks` | Start the missing bank, or use `--expected-banks` = number connected |
| `Address already in use` on port 9094 | Two banks on one laptop | Give the second one `--runtime-port 9095` |
| `ipconfig` shows several IPv4 addresses | Ethernet + Wi-Fi | Use the one under the *Wireless LAN adapter Wi-Fi* section |
| Each round takes ~25 s | Normal: Flower starts a fresh process per message on every bank | Just wait; 10 rounds take about 5 minutes |
