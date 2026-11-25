"""v12 step 1: assemble every number used in the revised manuscript (one LLM per provider; Sonnet results kept on disk but not reported).

Main study: 70 deployments, 15 districts, leave-one-district-out. Four LLM auditors of the same explained random
forest (Claude Opus 5.5, Claude Sonnet 5.5, GPT-6 Codex, Gemini 4 Hydrogen), plus the frozen v5 baselines.
External: Dongguan (v7, grouped five-fold) and Sheffield (v8, grouped three-fold, RF trained on Sheffield),
with the same four auditors (v10 for Opus). Nothing is refitted; all scores come from locked files.
The Opus main-study outputs are copied unchanged into derived/v11_final/llm_raw and hash-recorded.
"""
import os, sys, json, glob, hashlib, shutil
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE))
import v6_score_blind as V6
OUT = ROOT / "derived" / "v12_final"; (OUT / "llm_raw_zero_shot").mkdir(parents=True, exist_ok=True)
V11 = ROOT / "derived" / "v11_final"
SCR = ROOT / "llm_runs" / "opus_main"
BLIND = ROOT.parent / "Paper6_ChatGPT_blind_run"
KEYS = {"site", "perception", "temporal", "spatial", "ml_and_cases", "audit_adjustment", "audit_reason", "synthesis",
        "leak_probability", "verdict", "action", "work_order"}

# ---- Opus main-study audits are locked in v11_final; Opus zero-shot outputs are copied and locked here ----
assert json.load(open(V11 / "opus_main_locked.json")) == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((V11 / "llm_raw").glob("*.json"))}
ZSS = ROOT / "llm_runs" / "opus_zero_shot"
for f in sorted(ZSS.glob("claude-opus_Z_*.json")):
    dst = OUT / "llm_raw_zero_shot" / f.name
    if not dst.exists(): shutil.copy2(f, dst)
zh = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT / "llm_raw_zero_shot").glob("*.json"))}
lock = OUT / "opus_zero_shot_locked.json"
if lock.exists(): assert json.load(open(lock)) == zh, "locked zero-shot outputs changed"
else: json.dump(zh, open(lock, "w"), indent=1)

LLMS = {"Claude Opus 5.5": (V11 / "llm_raw", "claude-opus"),
        "GPT-6 Codex": (BLIND / "outputs", "chatgpt"), "Gemini 4 Hydrogen": (ROOT / "derived/v5/llm_raw", "gemini-4-hydrogen")}
site = pd.read_csv(ROOT / "derived/v6_blind_comparison/site_scores.csv")
dep = pd.read_csv(ROOT / "derived/v3/deployments_v3.csv").set_index("anon")
y = site.label.values; ctrl = site.is_negative_control.values.astype(bool); zone = site.zone.values; n = len(y)
rf = site["p_Random forest (tool used by H2)"].values
extra_keys = {}
recs = {}
for lab, (d, pre) in LLMS.items():
    rows = []
    for z in sorted(set(zone)):
        f = d / f"{pre}_H2_{z}.json"; arr = json.load(open(f, encoding="utf-8"))
        arr = arr["results"] if isinstance(arr, dict) else arr
        want = sorted(site.anon[site.zone == z]); assert sorted(r["site"] for r in arr) == want, (lab, z)
        for r in arr:
            assert not (KEYS - set(r)), (lab, r["site"]); assert int(r["audit_adjustment"]) in (-2, -1, 0, 1, 2)
            ex = set(r) - KEYS
            if ex: extra_keys.setdefault(lab, []).append((r["site"], sorted(ex)))
            rows.append({k: r[k] for k in KEYS})
    recs[lab] = {r["site"]: r for r in rows}
    a = site.anon.map(lambda s: int(recs[lab][s]["audit_adjustment"])).values
    site[f"a_{lab}"] = a; site[f"own_{lab}"] = site.anon.map(lambda s: float(recs[lab][s]["leak_probability"])).values
    site[f"p_{lab}"] = V6.primary(rf, a); site[f"act_{lab}"] = site.anon.map(lambda s: recs[lab][s]["action"]).values
    for k in ("synthesis", "audit_reason", "work_order", "verdict"): site[f"{k}_{lab}"] = site.anon.map(lambda s: recs[lab][s][k]).values
print("extra keys ignored:", extra_keys)

MODELS = {"Random forest, benchmark": "p_Random forest", "Random forest, audited tool": "p_Random forest (tool used by H2)"}
MODELS.update({f"Audited RF, {lab}": f"p_{lab}" for lab in LLMS})
for nm, c in [("CatBoost", "CatBoost"), ("k-nearest neighbours", "k-nearest neighbours"), ("Extra trees", "Extra trees"), ("XGBoost", "XGBoost"),
              ("Logistic regression", "Logistic regression"), ("Gradient boosting", "Gradient boosting"), ("Support vector machine", "Support vector machine"),
              ("LightGBM", "LightGBM"), ("Level-and-spread heuristic", "Level-and-spread heuristic"), ("Rule engine", "Rule engine (no LLM)")]:
    MODELS[nm] = "p_" + c
zs = {}
for f in sorted((OUT / "llm_raw_zero_shot").glob("claude-opus_Z_*.json")):
    arr = json.load(open(f, encoding="utf-8")); arr = arr["results"] if isinstance(arr, dict) else arr
    for r in arr: zs[r["site"]] = float(r["leak_probability"])
assert set(zs) == set(site.anon), "zero-shot outputs incomplete"
site["p_zs_opus"] = site.anon.map(zs).values; MODELS["Zero-shot LLM"] = "p_zs_opus"
MODELS.update({f"LLM own probability, {lab}": f"own_{lab}" for lab in LLMS})
idx = V6.bootstrap_indices(y, zone)
rows = []; boots = {}
for nm, c in MODELS.items():
    p = site[c].values.astype(float); bi = V6.bauc(y, p, idx["iid"]); bc = V6.bauc(y, p, idx["district_cluster"]); boots[nm] = (bi, bc)
    rows.append(dict(model=nm, **V6.measures(y, p, ctrl), AUC_lo=np.percentile(bi, 2.5), AUC_hi=np.percentile(bi, 97.5),
                     AUC_clo=np.percentile(bc, 2.5), AUC_chi=np.percentile(bc, 97.5)))
main = pd.DataFrame(rows); main.to_csv(OUT / "main_metrics.csv", index=False)
pairs = []
for lab in LLMS:
    k = f"Audited RF, {lab}"; p = site[f"p_{lab}"].values
    for ref in ["Random forest, audited tool", "Random forest, benchmark"]:
        q = site[MODELS[ref]].values; di = boots[k][0] - boots[ref][0]; dc = boots[k][1] - boots[ref][1]
        A = (p >= .5) == y; Bq = (q >= .5) == y; w, l = int((A & ~Bq).sum()), int((~A & Bq).sum())
        pairs.append(dict(model=k, reference=ref, dAUC=roc_auc_score(y, p) - roc_auc_score(y, q), lo=np.percentile(di, 2.5), hi=np.percentile(di, 97.5),
                          clo=np.percentile(dc, 2.5), chi=np.percentile(dc, 97.5), fixed=w, broken=l, mcnemar=binomtest(w, w + l, .5).pvalue if w + l else 1.0))
pd.DataFrame(pairs).to_csv(OUT / "main_paired.csv", index=False)
aud, tri = [], []
for lab in LLMS:
    a = site[f"a_{lab}"].values; oc = (rf >= .5) == y; nc = (site[f"p_{lab}"].values >= .5) == y
    aud.append(dict(model=lab, changed=int((a != 0).sum()), down_leak=int(((a < 0) & (y == 1)).sum()), down_sound=int(((a < 0) & (y == 0)).sum()),
                    up_leak=int(((a > 0) & (y == 1)).sum()), up_sound=int(((a > 0) & (y == 0)).sum()), fixed=int((~oc & nc).sum()), broken=int((oc & ~nc).sum()),
                    **{f"n_a{v:+d}": int((a == v).sum()) for v in (-2, -1, 1, 2)}))
    for act in ["DISPATCH EXCAVATION CREW", "EXTEND MONITORING", "NO EXCAVATION"]:
        m = site[f"act_{lab}"].values == act
        tri.append(dict(model=lab, action=act, n=int(m.sum()), leaking=int((m & (y == 1)).sum()), sound=int((m & (y == 0)).sum()), control=int((m & ctrl).sum())))
pd.DataFrame(aud).to_csv(OUT / "main_audit.csv", index=False); pd.DataFrame(tri).to_csv(OUT / "main_triage.csv", index=False)
# agreement among the four auditors
A = np.vstack([site[f"a_{lab}"].values for lab in LLMS]); D = np.vstack([site[f"p_{lab}"].values >= .5 for lab in LLMS])
agree = dict(all_zero=int((A == 0).all(0).sum()), any_nonzero=int((A != 0).any(0).sum()), same_decisions=int((D == D[0]).all(0).sum()),
             unanimous_action=int(np.all([site[f"act_{lab}"].values == site[f"act_{list(LLMS)[0]}"].values for lab in LLMS], axis=0).sum()))
# feature context for figures
for c in ["mean_leak_ratio", "persistence_ratio", "transient_ratio", "mean_centroid", "mean_rms", "mean_flatness", "n_nights" if "n_nights" in dep.columns else "mean_rms"]:
    site[c] = site.anon.map(dep[c])
site.to_csv(OUT / "site_level.csv", index=False)

# ---- external (Dongguan v7 grouped CV; Sheffield v8 grouped CV with Sheffield-trained RF) ----
ext = pd.read_csv(ROOT / "derived/v10_opus_external/metrics.csv")
ext = ext[ext.dataset.isin(["dongguan (v7 grouped CV)", "sheffield (v8 local grouped CV)"]) & (ext.model.str.endswith("| primary") | (ext.model == "RF tool")) & ~ext.model.str.startswith("claude-sonnet")].copy()
NM = {"RF tool": "Random forest", "claude-opus-5.5 | primary": "Audited RF, Claude Opus 5.5", "claude-sonnet-5.5 | primary": "Audited RF, Claude Sonnet 5.5",
      "codex-gpt-6.1-sol | primary": "Audited RF, GPT-6 Codex", "gemini-4-hydrogen | primary": "Audited RF, Gemini 4 Hydrogen"}
ext["model"] = ext.model.map(NM); ext["dataset"] = ext.dataset.map({"dongguan (v7 grouped CV)": "Dongguan", "sheffield (v8 local grouped CV)": "Sheffield"})
ext.to_csv(OUT / "external_metrics.csv", index=False)
eb = pd.read_csv(ROOT / "derived/v10_opus_external/audit_behavior.csv"); eb = eb[eb.dataset.isin(["dongguan (v7 grouped CV)", "sheffield (v8 local grouped CV)"]) & (eb.model != "claude-sonnet-5.5")].copy()
eb["dataset"] = eb.dataset.map({"dongguan (v7 grouped CV)": "Dongguan", "sheffield (v8 local grouped CV)": "Sheffield"})
eb["model"] = eb.model.map({"claude-opus-5.5": "Claude Opus 5.5", "claude-sonnet-5.5": "Claude Sonnet 5.5", "codex-gpt-6.1-sol": "GPT-6 Codex", "gemini-4-hydrogen": "Gemini 4 Hydrogen"})
eb.to_csv(OUT / "external_audit.csv", index=False)
ep = pd.read_csv(ROOT / "derived/v10_opus_external/paired.csv"); ep = ep[ep.reference == "RF tool"]; ep.to_csv(OUT / "external_paired_opus.csv", index=False)
# other models' paired vs RF from v7/v8
v7p = pd.read_csv(ROOT / "derived/v7_external_hybrids/paired_comparisons.csv"); v8p = pd.read_csv(ROOT / "derived/v8_sheffield_local/local_paired_comparisons.csv")
v7p = v7p[(v7p.dataset == "dongguan") & (v7p.reference == "RF tool") & ~v7p.model.str.startswith("claude-sonnet")]; v7p.to_csv(OUT / "external_paired_v7_dongguan.csv", index=False); v8p = v8p[~v8p.model.str.startswith("claude-sonnet")]; v8p.to_csv(OUT / "external_paired_v8_sheffield.csv", index=False)
# dataset descriptors
res = json.load(open(BLIND / "external_validation/results.json"))
desc = dict(dongguan=dict(n=res["dongguan"]["overall"]["n"], leak=res["dongguan"]["overall"]["n_leak"], noleak=res["dongguan"]["overall"]["n_no_leak"],
                          groups=res["dongguan"]["unique_groups"], raw=res["dongguan"]["raw_files"], dup=res["dongguan"]["exact_duplicate_files_removed"]),
            sheffield=dict(n=res["sheffield_zero_shot"]["overall"]["n"], leak=res["sheffield_zero_shot"]["overall"]["n_leak"],
                           noleak=res["sheffield_zero_shot"]["overall"]["n_no_leak"], files=res["sheffield_zero_shot"]["source_files"],
                           raw=res["sheffield_zero_shot"]["raw_channel_records"], dup=res["sheffield_zero_shot"]["exact_duplicate_channels_removed"]))
summary = dict(n=n, leaking=int(y.sum()), controls=int(ctrl.sum()), districts=len(set(zone)), agreement=agree, extra_keys=extra_keys, external=desc,
               features=res.get("features"))
json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=str)
pd.set_option("display.width", 250)
print(main[["model", "AUC", "AUC_clo", "AUC_chi", "accuracy", "precision", "recall", "F1", "specificity", "FP", "FN", "control_FA"]].round(3).to_string(index=False))
print(pd.DataFrame(pairs).round(3).to_string(index=False)); print(pd.DataFrame(aud).to_string(index=False)); print(pd.DataFrame(tri).to_string(index=False)); print(agree)
print(ext[["dataset", "model", "AUC", "AUC_cluster_lo", "AUC_cluster_hi", "accuracy", "precision", "sensitivity", "F1", "FP", "FN"]].round(3).to_string(index=False))
print(eb.to_string(index=False)); print(desc)
