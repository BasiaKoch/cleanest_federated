"""One-off extraction of the FedNova collapse evidence. Read-only over results/."""
import json, csv, glob, os, statistics
from collections import defaultdict

ROOT = "/Users/basiakoch/cleanest_federated/mnist_dermnist/results"
SEEDS = [42, 123, 456, 999, 2024, 31337, 271828, 161803, 789, 8675309]
COLLAPSED = {31337, 271828, 161803}
HEALTHY = {42, 123}

def load_json(p):
    with open(p) as f:
        return json.load(f)

def scan_test(dirpath, algo_tag):
    """Return {seed: dict(macro_f1, bal_acc, acc, per_class_f1, sel_round)}"""
    out = {}
    for p in glob.glob(os.path.join(dirpath, f"test_at_best_{algo_tag}*_s*.json")):
        d = load_json(p)
        s = int(d.get("seed"))
        out[s] = {
            "macro_f1": d.get("macro_f1"),
            "bal_acc": d.get("balanced_accuracy"),
            "acc": d.get("accuracy"),
            "pcf1": d.get("per_class_f1"),
            "sel_round": d.get("selected_round"),
            "best_val": d.get("best_val_macro_f1"),
            "partition": d.get("partition"),
            "momentum": d.get("momentum"),
            "lr": d.get("lr"),
            "num_rounds": d.get("num_rounds"),
        }
    return out

def table(title, data):
    print(f"\n===== {title} =====")
    print(f"{'seed':>9} {'macroF1':>8} {'balAcc':>7} {'acc':>6} {'selRnd':>6}  per_class_f1")
    for s in SEEDS:
        if s in data:
            d = data[s]
            tag = " <COLLAPSE?" if (d['macro_f1'] is not None and d['macro_f1'] < 0.4) else ""
            pcf = ",".join(f"{x:.2f}" for x in (d['pcf1'] or []))
            mf = f"{d['macro_f1']:.4f}" if d['macro_f1'] is not None else "  NA  "
            print(f"{s:>9} {mf:>8} {d['bal_acc']:.3f} {d['acc']:.3f} {str(d['sel_round']):>6}  [{pcf}]{tag}")
    vals = [data[s]['macro_f1'] for s in SEEDS if s in data and data[s]['macro_f1'] is not None]
    if vals:
        print(f"  n={len(vals)} mean={statistics.mean(vals):.4f} sd={statistics.pstdev(vals):.4f} "
              f"min={min(vals):.4f} max={max(vals):.4f}")

# --- Random straggler regime: the three algorithms ---
fednova_r = scan_test(f"{ROOT}/system_het_random_fednova", "fednova")
fedavg_r  = scan_test(f"{ROOT}/system_het_random", "fedavg")
fedprox_r = scan_test(f"{ROOT}/system_het_random", "fedprox")
table("RANDOM stragglers — FedNova", fednova_r)
table("RANDOM stragglers — FedAvg", fedavg_r)
table("RANDOM stragglers — FedProx", fedprox_r)

# --- Fixed straggler regime (only fedavg/fedprox have all seeds) ---
fedavg_f  = scan_test(f"{ROOT}/system_het_fixed", "fedavg")
fedprox_f = scan_test(f"{ROOT}/system_het_fixed", "fedprox")
table("FIXED stragglers — FedAvg", fedavg_f)
table("FIXED stragglers — FedProx", fedprox_f)

# Meta for fednova random
if fednova_r:
    any_s = next(iter(fednova_r.values()))
    print(f"\n[FedNova random meta] partition={any_s['partition']} momentum={any_s['momentum']} "
          f"lr={any_s['lr']} num_rounds={any_s['num_rounds']}")

print("\n\n##### COLLAPSE TRAJECTORY (FedNova random) — val_macro_f1 by round #####")
def read_history(path):
    rounds, vmf, vloss = [], [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rounds.append(int(row["round"]))
            vmf.append(float(row["val_macro_f1"]))
            vloss.append(float(row["val_loss"]))
    return rounds, vmf, vloss

for s in [42, 123, 31337, 271828, 161803]:
    p = f"{ROOT}/system_het_random_fednova/history_fednova_mu0.0_E20_sh-random_stragglers_s{s}.csv"
    if not os.path.exists(p):
        continue
    rnds, vmf, vloss = read_history(p)
    # sample every ~15 rounds + peak
    peak = max(vmf); peak_r = rnds[vmf.index(peak)]
    last = vmf[-1]; last_loss = vloss[-1]
    # find collapse: first round after peak where vmf drops below 0.25
    coll_r = None
    seen_peak = False
    for i, r in enumerate(rnds):
        if r >= peak_r: seen_peak = True
        if seen_peak and vmf[i] < 0.20:
            coll_r = r; break
    tag = "HEALTHY" if s in HEALTHY else ("COLLAPSED" if s in COLLAPSED else "")
    sparse = " ".join(f"r{rnds[i]}:{vmf[i]:.2f}" for i in range(0, len(rnds), max(1,len(rnds)//10)))
    print(f"\nseed {s} [{tag}]: peak_vmf={peak:.3f}@r{peak_r}  final_vmf={last:.3f} final_vloss={last_loss:.2f}"
          + (f"  COLLAPSE<0.20@r{coll_r}" if coll_r else "  (no <0.20 collapse)"))
    print(f"   {sparse}")
