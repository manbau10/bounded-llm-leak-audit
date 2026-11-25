"""v3 step 5: score all conditions on 70 deployments (15-district LODO) per PROTOCOL_v3.md.
Fast vectorised bootstrap (rank-based AUC)."""
import os, sys, json, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata, binomtest
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); V3 = os.path.join(ROOT, "derived", "v3")
d = pd.read_csv(os.path.join(V3, "deployments_v3.csv")); key = json.load(open(os.path.join(V3, "anon_key_v3.json")))
y = d.label.values; ctrl = d.is_negative_control.values.astype(bool); new = d.new_district.values.astype(bool); dist = d.district.values
FEATS = ["mean_rms", "median_rms", "mean_leak_ratio", "persistence_ratio", "stationarity_index", "transient_ratio", "mean_flatness", "mean_centroid", "mean_env_var"]
X = d[FEATS].values; P = {}

def lodo(make):
    p = np.zeros(len(d))
    for t in np.unique(dist):
        tr, te = dist != t, dist == t; sc = StandardScaler().fit(X[tr]); p[te] = make().fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]
    return p
P["Logistic regression"] = lodo(lambda: LogisticRegression(class_weight="balanced", max_iter=2000))
P["Support vector machine"] = lodo(lambda: SVC(probability=True, class_weight="balanced", random_state=0))
P["Random forest"] = lodo(lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42))
P["Gradient boosting"] = lodo(lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=42))
# --- Amendment 1 baselines (fixed settings, in-fold class balancing only) ---
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
def lodo_w(make):
    p = np.zeros(len(d))
    for t in np.unique(dist):
        tr, te = dist != t, dist == t; sc = StandardScaler().fit(X[tr]); ytr = y[tr]; spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        p[te] = make(spw).fit(sc.transform(X[tr]), ytr).predict_proba(sc.transform(X[te]))[:, 1]
    return p
P["XGBoost"] = lodo_w(lambda w: XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=w, random_state=42, verbosity=0, n_jobs=1))
P["LightGBM"] = lodo_w(lambda w: LGBMClassifier(n_estimators=200, num_leaves=8, learning_rate=0.05, min_child_samples=5, scale_pos_weight=w, random_state=42, verbose=-1, n_jobs=1))
P["CatBoost"] = lodo_w(lambda w: CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, scale_pos_weight=w, random_seed=42, verbose=0, thread_count=1))
P["Extra trees"] = lodo(lambda: ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=42))
P["k-nearest neighbours"] = lodo(lambda: KNeighborsClassifier(n_neighbors=5))
P["Neural network (MLP)"] = lodo(lambda: MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-2, max_iter=3000, random_state=42))
P["Level-and-spread heuristic"] = (1 / (1 + np.exp(-3.0 * ((d.mean_rms / 2500.0) * (2.5 / np.maximum(d.transient_ratio, 0.5)) - 1.0)))).values
def rule(r):
    a = (.45 if r.mean_leak_ratio >= .35 else 0) + (.25 if 400 <= r.mean_centroid <= 950 else 0) + (.10 if r.mean_flatness >= .10 else 0)
    t = (.40 if r.persistence_ratio >= .40 else (.15 if r.persistence_ratio >= .15 else 0)) - (.25 if r.transient_ratio > 2.5 else 0)
    return min(.99, max(.01, 1 / (1 + np.exp(-4.5 * (.5 * a + .5 * t - .25)))))
P["Deterministic rule engine (no LLM)"] = d.apply(rule, axis=1).values

# ---------- LLM outputs ----------
anon2idx = {a: i for i, a in enumerate(d.anon)}
def put(name, recs):
    p = np.full(len(d), np.nan)
    for r in recs: p[anon2idx[r["site"]]] = float(r["leak_probability"])
    P[name] = p
RAW = os.path.join(V3, "llm_raw"); ACT = {}
# Claude H (subagent files copied to llm_raw/claude_H_*.json), Claude Z new zones + v2 outputs for the original 12 districts
for pat, nm in [("claude_H_*.json", "Claude Sonnet 5.5 hybrid (H)"), ("opus_H_*.json", "Claude Opus 5.5 hybrid (H)")]:
    recs = []
    for f in sorted(glob.glob(os.path.join(RAW, pat))): recs += json.load(open(f))
    if len({r["site"] for r in recs}) == len(d): put(nm, recs); ACT[nm] = {r["site"]: r.get("action", "") for r in recs}
zrec = []
for f in sorted(glob.glob(os.path.join(RAW, "claude_Z_*.json"))): zrec += json.load(open(f))
v2key = json.load(open(os.path.join(ROOT, "derived", "v2", "anon_key.json")))["site"]; v2 = pd.read_csv(os.path.join(ROOT, "derived", "v2", "llm_outputs.csv")); v2 = v2[v2.condition == "naive"]
inv = {v: k for k, v in v2key.items()}
for r in v2.itertuples():
    dep = inv[r.site]; zrec.append(dict(site=d.loc[d.deploy_id == dep, "anon"].iloc[0], leak_probability=r.leak_probability))
if len(zrec) == len(d): put("Claude zero-shot (Z)", zrec)
for f in sorted(glob.glob(os.path.join(RAW, "g*_*_*.json"))):
    j = json.load(open(f)); tag = os.path.basename(f).split("_")[0]; cond = j["condition"]
    P.setdefault("_g_" + tag + "_" + cond, [])
    P["_g_" + tag + "_" + cond] = P["_g_" + tag + "_" + cond] + j["results"]
for k in [k for k in P if k.startswith("_g_")]:
    recs = P.pop(k); _, tag, cond = k.split("_", 2);
    if len({r["site"] for r in recs}) == len(d):
        nm = f"Gemini {cond}"; put(nm, recs); ACT[nm] = {r["site"]: r.get("action", "") for r in recs}

# ---------- metrics ----------
rng = np.random.default_rng(2026); B = 4000; n = len(y); IDX = rng.integers(0, n, (B, n))
def auc_boot(p):
    Pm, Ym = p[IDX], y[IDX]; rk = rankdata(Pm, axis=1); pos = Ym == 1; n1 = pos.sum(1); n0 = n - n1; ok = (n1 > 0) & (n0 > 0)
    return (((rk * pos).sum(1) - n1 * (n1 + 1) / 2) / np.where(ok, n1 * n0, 1))[ok]
def mets(p, pred=None):
    pr = (p >= .5).astype(int) if pred is None else pred
    tp = int(((pr == 1) & (y == 1)).sum()); fp = int(((pr == 1) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum()); tn = int(((pr == 0) & (y == 0)).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, acc=(tp + tn) / n, prec=tp / max(tp + fp, 1), rec=tp / (tp + fn), spec=tn / (tn + fp), f1=2 * tp / max(2 * tp + fp + fn, 1), nc_fa=int(((pr == 1) & ctrl).sum()))
rows = []
for nm, p in P.items():
    b = auc_boot(p); rows.append(dict(model=nm, auc=roc_auc_score(y, p), auc_lo=np.percentile(b, 2.5), auc_hi=np.percentile(b, 97.5), **mets(p)))
res = pd.DataFrame(rows); res.to_csv(os.path.join(V3, "benchmark_v3.csv"), index=False)
pd.set_option("display.width", 250); print(res.round(3).to_string(index=False))

rf = P["Random forest"]; print("\nPaired AUC difference vs random forest (model minus RF), 95% CI, one-sided p(diff<=0), non-inferiority P(diff>-0.05):")
for nm, p in P.items():
    if nm == "Random forest": continue
    # paired bootstrap on same indices
    Pa, Pb, Ym = p[IDX], rf[IDX], y[IDX]; pos = Ym == 1; n1 = pos.sum(1); n0 = n - n1; ok = (n1 > 0) & (n0 > 0)
    f = lambda M: (((rankdata(M, axis=1) * pos).sum(1) - n1 * (n1 + 1) / 2) / np.where(ok, n1 * n0, 1))[ok]
    dd = f(Pa) - f(Pb); obs = roc_auc_score(y, p) - roc_auc_score(y, rf)
    print(f"  {nm:38s} {obs:+.3f} ({np.percentile(dd,2.5):+.3f},{np.percentile(dd,97.5):+.3f}) p~{(dd<=0).mean():.3f} NI~{(dd>-0.05).mean():.3f}")
    a, c = (p >= .5).astype(int), (rf >= .5).astype(int); n01 = int(((a == y) & (c != y)).sum()); n10 = int(((a != y) & (c == y)).sum())
    print(f"       McNemar vs RF: model-correct-only {n01}, RF-correct-only {n10}, p={binomtest(n01, n01+n10, .5).pvalue if n01+n10 else 1:.3f}")
print("\nNew-district subset (10 deployments, 1 leaking) predictions p>=0.5 count / correct:")
for nm, p in P.items(): print(f"  {nm:38s} correct {int(((p[new] >= .5).astype(int) == y[new]).sum())}/{int(new.sum())}")
print("\nAction triage by truth:")
for nm, a in ACT.items():
    act = np.array([a[s] for s in d.anon]); print(nm); print(pd.crosstab(pd.Series(act, name="action"), pd.Series(y, name="label")))
out = d[["deploy_id", "district", "anon", "zone", "site_no", "label", "is_negative_control", "new_district", "n_nights", "mean_leak_ratio", "persistence_ratio", "transient_ratio", "mean_centroid"]].copy()
for nm, p in P.items(): out["p_" + nm] = p
for nm, a in ACT.items(): out["action_" + nm] = [a[s] for s in d.anon]
out.to_csv(os.path.join(V3, "predictions_v3.csv"), index=False)
json.dump({k: [list(map(float, roc_curve(y, v)[0])), list(map(float, roc_curve(y, v)[1]))] for k, v in P.items()}, open(os.path.join(V3, "roc_v3.json"), "w"))

# ---- Amendment 1: paired comparisons against the primary hybrid (Sonnet H) ----
REF = "Claude Sonnet 5.5 hybrid (H)"
if REF in P:
    hp = P[REF]; rows2 = []
    for nm, p in P.items():
        if nm == REF: continue
        Pa, Pb, Ym = hp[IDX], p[IDX], y[IDX]; pos = Ym == 1; n1 = pos.sum(1); n0 = n - n1; ok = (n1 > 0) & (n0 > 0)
        f = lambda M: (((rankdata(M, axis=1) * pos).sum(1) - n1 * (n1 + 1) / 2) / np.where(ok, n1 * n0, 1))[ok]
        dd = f(Pa) - f(Pb); a, c = (hp >= .5).astype(int), (p >= .5).astype(int)
        n01 = int(((a == y) & (c != y)).sum()); n10 = int(((a != y) & (c == y)).sum())
        rows2.append(dict(comparator=nm, auc_H_minus_model=roc_auc_score(y, hp) - roc_auc_score(y, p), lo=np.percentile(dd, 2.5), hi=np.percentile(dd, 97.5),
                          p_one_sided=(dd <= 0).mean(), H_correct_only=n01, model_correct_only=n10, mcnemar_p=binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.0))
    r2 = pd.DataFrame(rows2); r2.to_csv(os.path.join(V3, "paired_vs_hybrid_v3.csv"), index=False)
    print("\nPaired vs Sonnet hybrid (positive = hybrid better):"); print(r2.round(3).to_string(index=False))
