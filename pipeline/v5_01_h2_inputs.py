"""v5 (Amendment 3): H2 inputs. LLM audits an explained, in-fold random forest score."""
import os, json, copy, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
V3 = os.path.join(ROOT, "derived", "v3"); V5 = os.path.join(ROOT, "derived", "v5"); INP = os.path.join(V5, "inputs"); os.makedirs(INP, exist_ok=True)
d = pd.read_csv(os.path.join(V3, "deployments_v3.csv")); full = json.load(open(os.path.join(V3, "packets_full.json")))
dmap = json.load(open(os.path.join(V3, "anon_key_v3.json")))["district"]
FEATS = ["mean_rms", "median_rms", "mean_leak_ratio", "persistence_ratio", "stationarity_index", "transient_ratio", "mean_flatness", "mean_centroid", "mean_env_var"]
LABEL = {"mean_rms": "mean RMS", "median_rms": "median RMS", "mean_leak_ratio": "leak-band ratio", "persistence_ratio": "nightly persistence",
         "stationarity_index": "stationarity index", "transient_ratio": "transient ratio", "mean_flatness": "spectral flatness",
         "mean_centroid": "spectral centroid", "mean_env_var": "envelope variance"}

INSTR = (
 "You are the Chief Diagnostic Deliberation Agent in a multi-agent acoustic leak diagnosis system for urban water distribution mains. "
 "For each anonymised logger (site) the input gives: a random forest leak probability from a model trained only on other zones, that model's reliability table on other zones, "
 "a local explanation of which features pushed the score up or down, deterministic acoustic tool outputs, peer-site context in the same zone, five similar confirmed cases from other zones, "
 "reference tables, and a data-derived band profile. Your job is to AUDIT the random forest score, not to replace it. "
 "Judge every site independently. 1. Perception agent: interpret spectral evidence. 2. Temporal agent: interpret multi-night persistence versus transient disturbance. "
 "3. Spatial agent: interpret peer context (a leak is audible only over tens of metres; distances are unknown). "
 "4. Model agent: explain what drives the random forest score using the local explanation, and judge its reliability from the table. "
 "5. Case agent: compare the site with the similar confirmed cases and their outcomes. "
 "6. Chief auditor: decide whether the physical evidence agrees with the random forest score. Set audit_adjustment to 0 when the evidence agrees or is merely mixed. "
 "Use -1 or +1 only when there is a clear, specific physical reason that the score is too high (-1) or too low (+1). Use -2 or +2 only when several independent lines of evidence contradict the score. "
 "Most sites should receive 0. Also give your own calibrated probability that a leak lies within audible range, a verdict, a triage action, and a work order for a utility engineer. "
 "Background notes (general, not rules): (1) Leak noise from a pressurised pipe is broadband; its frequency content depends on pipe material, diameter, pressure, and leak size, so no single band defines a leak. "
 "(2) Leak noise attenuates along the pipe, faster in plastic pipes, so a leak may be quiet at a distant logger. (3) Traffic, domestic draw, and construction produce intermittent noise; a high transient ratio and low persistence suggest such sources. "
 "(4) Pumps, motors, transformers, and valves can produce steady tonal noise that persists at night and can mimic a leak; low flatness and narrow spread suggest a tonal source. "
 "(5) Persistence across many nights is stronger evidence than one night; records of only a few nights limit confidence. (6) Weak sensor coupling lowers amplitude. "
 "(7) A leak close to the logger can add energy at higher frequencies; judge it by flatness, persistence, and similar cases. (8) A model score is only as reliable as its record on other zones; peers in the same zone may share a source. "
 "Return ONLY a JSON array (no markdown fence, no commentary), one object per site, with exactly these keys: site, perception (max 30 words), temporal (max 30 words), spatial (max 25 words), "
 "ml_and_cases (max 35 words), audit_adjustment (integer -2, -1, 0, 1, or 2), audit_reason (max 30 words), synthesis (max 45 words), leak_probability (number 0 to 1), "
 "verdict (LEAK, NO LEAK, or AMBIGUOUS), action (DISPATCH EXCAVATION CREW, EXTEND MONITORING, or NO EXCAVATION), work_order (max 45 words).")

rf_p = {}
for dist, z in dmap.items():
    tr = d[d.district != dist].reset_index(drop=True); te = d[d.district == dist].reset_index(drop=True)
    m = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42).fit(tr[FEATS], tr.label)
    base = m.predict_proba(te[FEATS])[:, 1]; med = tr[FEATS].median()
    pk = copy.deepcopy(full[z])
    out = {"zone": z, "reference_from_other_zones": pk["reference_from_other_zones"], "data_derived_band_profile": pk["data_derived_band_profile"],
           "random_forest_reliability_on_other_zones": pk["rf_reliability_on_other_zones"], "sites": []}
    bysite = {s["site"]: s for s in pk["sites"]}
    for i, r in te.iterrows():
        contrib = []
        for f in FEATS:
            x = te.loc[[i], FEATS].copy(); x[f] = med[f]
            contrib.append((f, float(base[i] - m.predict_proba(x)[:, 1][0])))
        contrib.sort(key=lambda t: -abs(t[1]))
        expl = [{"feature": LABEL[f], "site_value": round(float(r[f]), 3), "training_median": round(float(med[f]), 3),
                 "effect_on_probability": round(c, 3)} for f, c in contrib[:4]]
        s = bysite[r.anon]; rf_p[r.anon] = float(base[i])
        out["sites"].append({"site": r.anon, "random_forest": {"leak_probability": round(float(base[i]), 2), "local_explanation_top_features": expl},
                             "perception_tool_output": s["perception_tool_output"], "temporal_tool_output": s["temporal_tool_output"],
                             "spatial_tool_output": s["spatial_tool_output"], "similar_cases_from_other_zones": s["similar_cases_from_other_zones"]})
    out["sites"].sort(key=lambda s: [x["site"] for x in pk["sites"]].index(s["site"]))   # keep the shuffled v3 order
    json.dump(dict(instructions=INSTR, input=out), open(os.path.join(INP, f"H2_{z}.json"), "w"), separators=(",", ":"))
json.dump(rf_p, open(os.path.join(V5, "rf_tool_probabilities.json"), "w"))
# consistency check against the v3 RF tool
old = {s["site"]: s["ml_tool_output"]["random_forest_leak_probability"] for z in full.values() for s in z["sites"]}
print("files:", len(os.listdir(INP)), "| max |RF v5 - RF v3 tool|:", round(max(abs(round(rf_p[k], 2) - old[k]) for k in rf_p), 3))
ex = json.load(open(os.path.join(INP, "H2_C.json")))["input"]["sites"][0]; print(json.dumps(ex["random_forest"], indent=1))
