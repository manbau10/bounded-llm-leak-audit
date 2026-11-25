import json
"""v3 step 2: deployment features (15 districts) and blind packets for conditions Z, T, T+ML, T+RAG, H.

Everything computed for a held-out district uses only the other 14 districts (reference tables, band profile,
RF tool, RF reliability table, nearest-neighbour retrieval). No labels, control flags, distances, district names
or logger IDs appear in any packet.
"""
import os, json, random, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "derived", "v3"); LEAK_THR = 0.35

f = pd.read_csv(os.path.join(OUT, "features_v3.csv")); f = f[f.quality_keep == 1].copy()
f["base_logger"] = f.logger_id.astype(str).str.replace(r"[a-zA-Z]", "", regex=True)
f["deploy_id"] = f.district + "_L" + f.base_logger
for b in ["0_200", "200_600", "600_1000", "1000_1500", "1500_2048"]:
    f["fr_" + b] = f["energy_" + b] / f["total_psd"]
rows = []
for dep, g in f.groupby("deploy_id"):
    nights = g.groupby("rec_date")["leak_band_ratio"].median(); rms = g.rms_amplitude.values
    rows.append(dict(deploy_id=dep, district=g.district.iloc[0], base_logger=g.base_logger.iloc[0],
        label=int((g.label == 1).sum() > (g.label == 0).sum()), is_negative_control=int(g.is_negative_control.iloc[0]),
        n_recordings=len(g), n_nights=int(nights.size), mean_rms=g.rms_amplitude.mean(), median_rms=float(np.median(rms)),
        mean_leak_ratio=float(g.leak_band_ratio.mean()), persistence_ratio=float((nights >= LEAK_THR).mean()),
        transient_ratio=float(np.percentile(rms, 95) / (np.median(rms) + 1e-9)),
        stationarity_index=float(g.leak_band_ratio.mean() / (g.leak_band_ratio.std(ddof=0) + 1e-9)),
        mean_flatness=g.spectral_flatness.mean(), mean_centroid=g.spectral_centroid.mean(), mean_spread=g.spectral_spread.mean(),
        mean_env_var=g.envelope_variance.median(), fr_b1=g["fr_0_200"].mean(), fr_b2=g["fr_200_600"].mean(), fr_b3=g["fr_600_1000"].mean(),
        fr_b4=g["fr_1000_1500"].mean(), fr_b5=g["fr_1500_2048"].mean()))
d = pd.DataFrame(rows).sort_values("deploy_id").reset_index(drop=True)
FEATS = ["mean_rms", "median_rms", "mean_leak_ratio", "persistence_ratio", "stationarity_index", "transient_ratio", "mean_flatness", "mean_centroid", "mean_env_var"]
BANDS = [("fr_b1", "0-200 Hz"), ("fr_b2", "200-600 Hz"), ("fr_b3", "600-1000 Hz"), ("fr_b4", "1000-1500 Hz"), ("fr_b5", "1500-2048 Hz")]

# ---- anonymisation (v3 key) ----
rng = random.Random(2026)
dists = sorted(d.district.unique()); letters = list("ABCDEFGHIJKLMNO"); rng.shuffle(letters); dmap = dict(zip(dists, letters))
smap = {}
for dist in dists:
    ids = list(d[d.district == dist].deploy_id); rng.shuffle(ids)
    for k, i in enumerate(ids, 1): smap[i] = f"{dmap[dist]}{k}"
d["anon"] = d.deploy_id.map(smap); d["zone"] = d.district.map(dmap)
site_no = {z: i + 1 for i, z in enumerate(sorted(dmap.values()))}
d["site_no"] = d.zone.map(site_no)
d["new_district"] = d.district.isin(json.load(open(os.path.join(ROOT, "private", "sites.json")))["added_sites"]).astype(int)  # private file, not in repo
d.to_csv(os.path.join(OUT, "deployments_v3.csv"), index=False)
json.dump(dict(district=dmap, site=smap, site_no=site_no), open(os.path.join(OUT, "anon_key_v3.json"), "w"), indent=1)
print(len(d), "deployments;", int(d.label.sum()), "leaking;", int(d.is_negative_control.sum()), "controls;", d.district.nunique(), "districts")

r3 = lambda x, n=3: round(float(x), n)
def rf(): return RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
REFC = [("mean_leak_ratio", "leak-band ratio"), ("mean_centroid", "spectral centroid (Hz)"), ("persistence_ratio", "nightly persistence"),
        ("transient_ratio", "transient ratio"), ("mean_flatness", "spectral flatness"), ("mean_rms", "mean RMS (counts)")]
def ref_table(tr):
    o = {}
    for lab, nm in [(1, "leaking"), (0, "sound")]:
        s = tr[tr.label == lab]; o[nm] = {"n_deployments": int(len(s))}
        for c, cn in REFC:
            q = np.percentile(s[c], [25, 50, 75]); o[nm][cn] = {"median": r3(q[1]), "IQR": [r3(q[0]), r3(q[2])]}
    return o
def band_profile(tr):
    o = {}
    for c, nm in BANDS:
        a, b = tr[tr.label == 1][c], tr[tr.label == 0][c]
        o[nm] = {"median_fraction_leaking": r3(a.median()), "median_fraction_sound": r3(b.median()), "single_band_AUC": r3(roc_auc_score(tr.label, tr[c]), 2)}
    return o
def inner_reliability(tr):
    p = np.zeros(len(tr)); trd = tr.district.values
    for t in np.unique(trd):
        m = trd == t; mdl = rf().fit(tr.loc[~m, FEATS], tr.label[~m]); p[m] = mdl.predict_proba(tr.loc[m, FEATS])[:, 1]
    y = tr.label.values; bins = [0, .2, .4, .6, .8, 1.0001]; tab = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi); tab.append({"rf_probability_bin": f"{lo:.1f}-{min(hi,1):.1f}", "n_deployments": int(m.sum()), "fraction_truly_leaking": (r3(y[m].mean(), 2) if m.sum() else None)})
    return {"inner_leave_one_zone_out_AUC": r3(roc_auc_score(y, p), 2), "inner_accuracy": r3(((p >= .5) == y).mean(), 2), "reliability_table": tab}
def sd(x): return x.astype(float)

pk = {}
for dist in dists:
    test = d[d.district == dist].copy(); tr = d[d.district != dist].reset_index(drop=True)
    mdl = rf().fit(tr[FEATS], tr.label); ptest = mdl.predict_proba(test[FEATS])[:, 1]
    sc = StandardScaler().fit(tr[FEATS]); Xtr = sc.transform(tr[FEATS]); Xte = sc.transform(test[FEATS])
    sites = []
    for i, (_, s) in enumerate(test.iterrows()):
        peers = test[test.deploy_id != s.deploy_id]
        peer_like = int(((peers.mean_leak_ratio >= LEAK_THR) & (peers.persistence_ratio >= 0.5)).sum())
        dist_ = np.linalg.norm(Xtr - Xte[i], axis=1); nn = np.argsort(dist_)[:5]
        cases = [dict(outcome="leaking" if tr.label[j] == 1 else "sound", distance=r3(dist_[j], 2), leak_band_ratio=r3(tr.mean_leak_ratio[j]),
                      persistence=r3(tr.persistence_ratio[j]), transient_ratio=r3(tr.transient_ratio[j], 2), centroid_Hz=r3(tr.mean_centroid[j], 0),
                      mean_rms=r3(tr.mean_rms[j], 0), n_nights=int(tr.n_nights[j])) for j in nn]
        sites.append(dict(site=s.anon,
            perception_tool_output=dict(n_recordings=int(s.n_recordings), mean_rms_counts=r3(s.mean_rms, 1),
                band_energy_fractions={nm: r3(s[c]) for c, nm in BANDS}, leak_band_ratio_200_1000Hz=r3(s.mean_leak_ratio),
                spectral_centroid_Hz=r3(s.mean_centroid, 1), spectral_spread_Hz=r3(s.mean_spread, 1), wiener_spectral_flatness=r3(s.mean_flatness, 4),
                envelope_variance_median=r3(s.mean_env_var)),
            temporal_tool_output=dict(n_nights=int(s.n_nights), nightly_persistence_ratio=r3(s.persistence_ratio),
                transient_ratio_p95_over_median_rms=r3(s.transient_ratio, 2), stationarity_index=r3(s.stationarity_index, 2)),
            spatial_tool_output=dict(n_peer_sites_in_same_zone=int(len(peers)), n_peer_sites_with_leak_like_signature=peer_like,
                note="Peer sites share the same zone. Distances to any leak are unknown at survey time."),
            ml_tool_output=dict(random_forest_leak_probability=r3(ptest[i], 2)),
            similar_cases_from_other_zones=cases,
            _zs=dict(mean_rms=r3(s.mean_rms, 1), median_rms=r3(s.median_rms, 1), mean_leak_band_ratio=r3(s.mean_leak_ratio), persistence_ratio=r3(s.persistence_ratio),
                     stationarity_index=r3(s.stationarity_index, 2), transient_ratio=r3(s.transient_ratio, 2), mean_spectral_flatness=r3(s.mean_flatness, 4),
                     mean_spectral_centroid_Hz=r3(s.mean_centroid, 1), median_envelope_variance=r3(s.mean_env_var))))
    pk[dmap[dist]] = dict(zone=dmap[dist], reference_from_other_zones=ref_table(tr), data_derived_band_profile=band_profile(tr),
                          rf_reliability_on_other_zones=inner_reliability(tr), sites=sites)
    print("fold", dmap[dist], dist, len(test), "RF inner AUC", pk[dmap[dist]]["rf_reliability_on_other_zones"]["inner_leave_one_zone_out_AUC"])
json.dump(pk, open(os.path.join(OUT, "packets_full.json"), "w"))
print("saved packets_full.json")
