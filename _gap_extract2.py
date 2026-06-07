import json, csv, glob, os, statistics
ROOT = "/Users/basiakoch/cleanest_federated/mnist_dermnist/results"

def lj(p):
    with open(p) as f: return json.load(f)

print("===== FedNova FIXED stragglers (fednova_unequal_E, seed 42 only) =====")
for lvl in ["L3_two_client_70_30_rare_enriched", "L4_two_client_90_10_rare_stress"]:
    d = f"{ROOT}/fednova_unequal_E/{lvl}"
    print(f"\n-- {lvl} --")
    for algo in ["fedavg_mu0.0", "fedprox_mu0.01", "fednova_mu0.0"]:
        for sh in ["", "_sh-fixed_stragglers"]:
            p = f"{d}/test_at_best_{algo}_E20{sh}_s42.json"
            if os.path.exists(p):
                j = lj(p)
                pcf = ",".join(f"{x:.2f}" for x in j.get("per_class_f1", []))
                shtag = "fixed-strag" if sh else "uniform    "
                print(f"  {algo:14s} {shtag}: macroF1={j['macro_f1']:.4f} balAcc={j['balanced_accuracy']:.3f} selRnd={j.get('selected_round')} pcf1=[{pcf}]")

print("\n\n===== FedNova UNIFORM baseline sanity (does it match FedAvg under no-straggler?) =====")
# heterogeneity_ladder L4 + headline + flower baseline
for d, label in [(f"{ROOT}/heterogeneity_ladder/L4_two_client_90_10_rare_stress", "ladder L4 90/10"),
                 (f"{ROOT}/headline", "headline")]:
    if not os.path.isdir(d): continue
    print(f"\n-- {label} ({d.split('/')[-1]}) --")
    for p in sorted(glob.glob(f"{d}/test_at_best_*_s42.json")):
        j = lj(p); algo = os.path.basename(p)
        print(f"  {os.path.basename(p)[13:-9]:40s}: macroF1={j['macro_f1']:.4f} part={j.get('partition')}")

print("\n\n===== UPDATE-NORM EVIDENCE: FedNova random, collapsed vs healthy =====")
def normstats(seed):
    p = f"{ROOT}/system_het_random_fednova/client_update_norms_fednova_mu0.0_E20_sh-random_stragglers_s{seed}.csv"
    if not os.path.exists(p): return None
    rows = list(csv.DictReader(open(p)))
    # columns: round, client_id, update_norm, n_samples, local_epochs, tau
    by_round_early = [r for r in rows if int(r["round"]) <= 5]
    norms = [float(r["update_norm"]) for r in rows]
    taus = [int(r["tau"]) for r in rows]
    # find rows with smallest tau and their norms
    small_tau = sorted(rows, key=lambda r: int(r["tau"]))[:5]
    big_tau = sorted(rows, key=lambda r: -int(r["tau"]))[:5]
    return {
        "cols": list(rows[0].keys()),
        "n": len(rows),
        "tau_range": (min(taus), max(taus)),
        "norm_range": (min(norms), max(norms)),
        "norm_mean": statistics.mean(norms),
        "norm_max": max(norms),
        "early_max_norm": max(float(r["update_norm"]) for r in by_round_early) if by_round_early else None,
        "small_tau_rows": [(int(r["round"]), int(r["client_id"]), int(r["tau"]), round(float(r["update_norm"]),3)) for r in small_tau],
        "big_tau_rows": [(int(r["round"]), int(r["client_id"]), int(r["tau"]), round(float(r["update_norm"]),3)) for r in big_tau],
    }
for s in [42, 123, 31337, 271828, 161803]:
    st = normstats(s)
    if st is None:
        print(f" seed {s}: no file"); continue
    print(f"\n seed {s}: cols={st['cols']}")
    print(f"   n_rows={st['n']} tau_range={st['tau_range']} norm_range=({st['norm_range'][0]:.3f},{st['norm_range'][1]:.3f}) norm_mean={st['norm_mean']:.3f} early(r<=5)_max_norm={st['early_max_norm']}")
    print(f"   smallest-tau rows (rnd,cid,tau,norm): {st['small_tau_rows']}")
    print(f"   biggest-tau  rows (rnd,cid,tau,norm): {st['big_tau_rows']}")

print("\n\n===== Round-1 tau distribution per seed (what makes 31337/271828/161803 special?) =====")
for s in [42,123,456,31337,271828,161803,789,8675309]:
    p = f"{ROOT}/system_het_random_fednova/client_update_norms_fednova_mu0.0_E20_sh-random_stragglers_s{s}.csv"
    if not os.path.exists(p): continue
    rows = [r for r in csv.DictReader(open(p)) if int(r["round"])==1]
    if not rows: continue
    info = sorted([(int(r["client_id"]), int(r["tau"]), int(r["local_epochs"]), round(float(r["update_norm"]),2)) for r in rows])
    taus = [t for _,t,_,_ in info]
    print(f" seed {s}: round1 (cid,tau,E,norm)={info}  tau_min={min(taus)} tau_max={max(taus)} ratio={max(taus)/max(1,min(taus)):.1f}")
