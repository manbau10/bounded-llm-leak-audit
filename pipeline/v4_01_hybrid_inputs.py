"""v4 (Amendment 2): hybrid H with a nested-selected, non-random-forest ML tool. No MLP anywhere.
Per outer fold, 8 candidates are ranked by inner leave-one-district-out AUC on the 14 training districts only."""
import os, json, copy, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
V3 = os.path.join(ROOT, "derived", "v3"); V4 = os.path.join(ROOT, "derived", "v4"); INP = os.path.join(V4, "inputs"); os.makedirs(INP, exist_ok=True)
d = pd.read_csv(os.path.join(V3, "deployments_v3.csv")); full = json.load(open(os.path.join(V3, "packets_full.json")))
prompts = json.load(open(os.path.join(V3, "prompts_v3.json")))
FEATS = ["mean_rms", "median_rms", "mean_leak_ratio", "persistence_ratio", "stationarity_index", "transient_ratio", "mean_flatness", "mean_centroid", "mean_env_var"]
CANDS = {
 "logistic regression": lambda w: LogisticRegression(class_weight="balanced", max_iter=2000),
 "support vector machine": lambda w: SVC(probability=True, class_weight="balanced", random_state=0),
 "gradient boosting": lambda w: HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
 "XGBoost": lambda w: XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=w, random_state=42, verbosity=0, n_jobs=1),
 "LightGBM": lambda w: LGBMClassifier(n_estimators=200, num_leaves=8, learning_rate=0.05, min_child_samples=5, scale_pos_weight=w, random_state=42, verbose=-1, n_jobs=1),
 "CatBoost": lambda w: CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, scale_pos_weight=w, random_seed=42, verbose=0, thread_count=1),
 "extra trees": lambda w: ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=42),
 "k-nearest neighbours": lambda w: KNeighborsClassifier(n_neighbors=5),
}
def fitpred(name, Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    return CANDS[name](w).fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]
def inner(name, tr):
    p = np.zeros(len(tr)); dd = tr.district.values; X = tr[FEATS].values; y = tr.label.values
    for t in np.unique(dd):
        m = dd == t; p[m] = fitpred(name, X[~m], y[~m], X[m])
    return p

r3 = lambda x, n=2: round(float(x), n)
dmap = json.load(open(os.path.join(V3, "anon_key_v3.json")))["district"]
sel_log, tool_p = [], {}
for dist, z in dmap.items():
    tr = d[d.district != dist].reset_index(drop=True); te = d[d.district == dist]
    inn = {nm: inner(nm, tr) for nm in CANDS}; aucs = {nm: roc_auc_score(tr.label, p) for nm, p in inn.items()}
    best = max(aucs, key=aucs.get); pin = inn[best]; y = tr.label.values
    pte = fitpred(best, tr[FEATS].values, y, te[FEATS].values)
    bins = [0, .2, .4, .6, .8, 1.0001]; tab = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pin >= lo) & (pin < hi); tab.append({"probability_bin": f"{lo:.1f}-{min(hi,1):.1f}", "n_deployments": int(m.sum()), "fraction_truly_leaking": (r3(y[m].mean()) if m.sum() else None)})
    rel = {"model": best, "inner_leave_one_zone_out_AUC": r3(aucs[best]), "inner_accuracy": r3(((pin >= .5) == y).mean()), "reliability_table": tab}
    sel_log.append(dict(zone=z, district=dist, selected=best, **{f"innerAUC_{k}": r3(v, 3) for k, v in aucs.items()}))
    pk = copy.deepcopy(full[z]); out = {"zone": z, "reference_from_other_zones": pk["reference_from_other_zones"], "data_derived_band_profile": pk["data_derived_band_profile"],
                                        "ml_model_reliability_on_other_zones": rel, "sites": []}
    for s, p in zip(pk["sites"], pte):
        tool_p[s["site"]] = float(p)
        out["sites"].append({"site": s["site"], "perception_tool_output": s["perception_tool_output"], "temporal_tool_output": s["temporal_tool_output"],
                             "spatial_tool_output": s["spatial_tool_output"], "ml_tool_output": {"model": best, "leak_probability": r3(p)},
                             "similar_cases_from_other_zones": s["similar_cases_from_other_zones"]})
    instr = (prompts["H"].replace("4. Model agent: interpret the random forest probability for the site", "4. Model agent: interpret the machine-learning model probability for the site")
             .replace("The random forest was trained only on other zones; its reliability table shows", "The machine-learning model named in the input was selected and trained only on other zones; its reliability table shows"))
    assert "random forest" not in instr.lower()
    json.dump(dict(instructions=instr, input=out), open(os.path.join(INP, f"H_{z}.json"), "w"), separators=(",", ":"))
    print(z, dist[:28].ljust(28), "->", best, round(aucs[best], 3))
pd.DataFrame(sel_log).to_csv(os.path.join(V4, "ml_tool_selection.csv"), index=False)
json.dump(tool_p, open(os.path.join(V4, "ml_tool_probabilities.json"), "w"))
txt = "".join(open(os.path.join(INP, f), encoding="utf-8").read() for f in os.listdir(INP))
print("random forest mentions:", txt.lower().count("random forest"), "| MLP mentions:", txt.lower().count("mlp"), "| files:", len(os.listdir(INP)))
