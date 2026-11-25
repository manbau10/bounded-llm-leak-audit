"""v5 (Amendment 3) scoring: RF restored, MLP excluded. H2 primary score = sigmoid(logit(p_RF) + 0.7*a)."""
import os, json, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata, binomtest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
V3, V4, V5 = (os.path.join(ROOT, "derived", v) for v in ("v3", "v4", "v5"))
d = pd.read_csv(os.path.join(V3, "deployments_v3.csv")); y = d.label.values; n = len(y); ctrl = d.is_negative_control.values.astype(bool); dist = d.district.values
FEATS = ["mean_rms", "median_rms", "mean_leak_ratio", "persistence_ratio", "stationarity_index", "transient_ratio", "mean_flatness", "mean_centroid", "mean_env_var"]
X = d[FEATS].values
src = open(os.path.join(HERE, "v4_01_hybrid_inputs.py"), encoding="utf-8").read(); ns = {}
exec(src.split("HERE = ")[0], ns); exec("CANDS = {" + src.split("CANDS = {")[1].split("}\ndef fitpred")[0] + "}", ns); CANDS = ns["CANDS"]
CANDS["random forest"] = lambda w: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
NAMES = {"random forest": "Random forest", "logistic regression": "Logistic regression", "support vector machine": "Support vector machine", "gradient boosting": "Gradient boosting",
         "XGBoost": "XGBoost", "LightGBM": "LightGBM", "CatBoost": "CatBoost", "extra trees": "Extra trees", "k-nearest neighbours": "k-nearest neighbours"}
P = {}
for k, nm in NAMES.items():
    p = np.zeros(n)
    for t in np.unique(dist):
        tr, te = dist != t, dist == t; sc = StandardScaler().fit(X[tr]); w = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        p[te] = CANDS[k](w).fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]
    P[nm] = p
P["Level-and-spread heuristic"] = (1 / (1 + np.exp(-3.0 * ((d.mean_rms / 2500.0) * (2.5 / np.maximum(d.transient_ratio, 0.5)) - 1.0)))).values
def rule(r):
    a = (.45 if r.mean_leak_ratio >= .35 else 0) + (.25 if 400 <= r.mean_centroid <= 950 else 0) + (.10 if r.mean_flatness >= .10 else 0)
    t = (.40 if r.persistence_ratio >= .40 else (.15 if r.persistence_ratio >= .15 else 0)) - (.25 if r.transient_ratio > 2.5 else 0)
    return min(.99, max(.01, 1 / (1 + np.exp(-4.5 * (.5 * a + .5 * t - .25)))))
P["Rule engine (no LLM)"] = d.apply(rule, axis=1).values
pv3 = pd.read_csv(os.path.join(V3, "predictions_v3.csv")).set_index("anon")
P["Claude zero-shot"] = d.anon.map(pv3["p_Claude zero-shot (Z)"]).values
P["Claude hybrid v3 (free probability)"] = d.anon.map(pv3["p_Claude Sonnet 5.5 hybrid (H)"]).values

rfp = d.anon.map(json.load(open(os.path.join(V5, "rf_tool_probabilities.json")))).values
print("max |RF tool - RF baseline|:", round(float(np.abs(rfp - P["Random forest"]).max()), 3), "| AUC tool", round(roc_auc_score(y, rfp), 3), "baseline", round(roc_auc_score(y, P["Random forest"]), 3)); P["Random forest (tool used by H2)"] = rfp
lg = lambda p: np.log(np.clip(p, .02, .98) / (1 - np.clip(p, .02, .98)))
AUD, ACT, TXT = {}, {}, {}
for pat, lab in [("claude-sonnet_H2_*.json", "Claude Sonnet 5.5"), ("gemini-4-hydrogen_H2_*.json", "Gemini 4 Hydrogen"), ("chatgpt_H2_*.json", "ChatGPT")]:
    recs = []
    for f in sorted(glob.glob(os.path.join(V5, "llm_raw", pat))):
        j = json.load(open(f)); recs += j["results"] if isinstance(j, dict) else j
    if len({r["site"] for r in recs}) != n: print("incomplete:", lab, len({r['site'] for r in recs})); continue
    m = {r["site"]: r for r in recs}; a = np.array([int(m[s]["audit_adjustment"]) for s in d.anon]); own = np.array([float(m[s]["leak_probability"]) for s in d.anon])
    P[f"H2 {lab} (primary: audited RF)"] = 1 / (1 + np.exp(-(lg(rfp) + 0.7 * a)))
    P[f"H2 {lab} (secondary: mean of RF and LLM)"] = (rfp + own) / 2
    P[f"H2 {lab} (LLM own probability)"] = own
    AUD[lab] = a; ACT[lab] = np.array([m[s].get("action", "") for s in d.anon]); TXT[lab] = m

rng = np.random.default_rng(2026); B = 4000; IDX = rng.integers(0, n, (B, n)); pos = y[IDX] == 1; n1 = pos.sum(1); ok = (n1 > 0) & (n1 < n)
def bauc(p): return (((rankdata(p[IDX], axis=1) * pos).sum(1) - n1 * (n1 + 1) / 2) / np.where(ok, n1 * (n - n1), 1))[ok]
def mets(p):
    pr = (p >= .5).astype(int); tp = int(((pr == 1) & (y == 1)).sum()); fp = int(((pr == 1) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum()); tn = n - tp - fp - fn
    prec = tp / max(tp + fp, 1); rec = tp / (tp + fn)
    return dict(accuracy=(tp + tn) / n, precision=prec, recall=rec, f1=(2 * prec * rec / (prec + rec) if prec + rec else 0), specificity=tn / (tn + fp), FP=fp, FN=fn, control_FA=int(((pr == 1) & ctrl).sum()))
rows = []
for nm, p in P.items():
    b = bauc(p); rows.append(dict(model=nm, AUC=roc_auc_score(y, p), AUC_lo=np.percentile(b, 2.5), AUC_hi=np.percentile(b, 97.5), **mets(p)))
res = pd.DataFrame(rows).sort_values("AUC", ascending=False); res.to_csv(os.path.join(V5, "benchmark_v5.csv"), index=False)
pd.set_option("display.width", 260); print(res.round(3).to_string(index=False))

rf = P["Random forest"]; print("\nPaired vs random forest (model minus RF; one-sided p = P(diff<=0); unadjusted):")
for nm, p in P.items():
    if not nm.startswith("H2") and "hybrid" not in nm: continue
    dd = bauc(p) - bauc(rf); a_, c_ = (p >= .5).astype(int), (rf >= .5).astype(int)
    n01 = int(((a_ == y) & (c_ != y)).sum()); n10 = int(((a_ != y) & (c_ == y)).sum())
    print(f"  {nm:48s} dAUC {roc_auc_score(y,p)-roc_auc_score(y,rf):+.3f} ({np.percentile(dd,2.5):+.3f},{np.percentile(dd,97.5):+.3f}) p={(dd<=0).mean():.3f} | right-only {n01} vs RF {n10}, McNemar p={binomtest(n01,n01+n10,.5).pvalue if n01+n10 else 1:.3f}")
for lab, a in AUD.items():
    print(f"\nAudit adjustments, {lab}: counts", dict(zip(*np.unique(a, return_counts=True))))
    for v in (-2, -1, 1, 2):
        m = a == v
        if m.sum(): print(f"   a={v:+d}: n={int(m.sum())}, truly leaking {int(y[m].sum())}, RF said leak {int((rf[m]>=.5).sum())}")
    print("   triage:"); print(pd.crosstab(pd.Series(ACT[lab], name="action"), pd.Series(y, name="truth")))
out = d[["deploy_id", "anon", "zone", "site_no", "label", "is_negative_control"]].copy()
for nm, p in P.items(): out["p_" + nm] = p
for lab, a in AUD.items(): out["audit_" + lab] = a; out["action_" + lab] = ACT[lab]
out.to_csv(os.path.join(V5, "predictions_v5.csv"), index=False)
