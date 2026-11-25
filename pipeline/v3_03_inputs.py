import json
"""v3 step 3: build per-condition, per-zone input files (instructions + blind packet). Frozen per PROTOCOL_v3.md."""
import os, json, copy

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
V3 = os.path.join(ROOT, "derived", "v3"); INP = os.path.join(V3, "inputs"); os.makedirs(INP, exist_ok=True)
full = json.load(open(os.path.join(V3, "packets_full.json")))

NOTES = [
 "Leak noise from a pressurised pipe is broadband. Its frequency content depends on pipe material, diameter, pressure, and leak size, so no single frequency band defines a leak.",
 "Leak noise attenuates along the pipe. Plastic pipes attenuate it faster than metallic pipes, so a leak may be quiet at a distant logger.",
 "Traffic, domestic draw, and construction produce intermittent noise. A high transient ratio and low nightly persistence suggest such sources.",
 "Pumps, motors, transformers, and valves can produce steady tonal noise that also persists at night and can mimic a persistent leak. Low spectral flatness and a narrow spread suggest a tonal source.",
 "Persistence across many nights is stronger evidence than energy on a single night. Records of only a few nights limit confidence.",
 "Weak sensor coupling lowers amplitude, so low RMS with an otherwise leak-like spectrum can reflect a distant leak or poor contact.",
 "A leak close to the logger can add energy at higher frequencies. Judge high-frequency energy by its flatness and persistence, and compare it with similar past cases.",
 "A statistical model score is only as reliable as its record on other zones. Use the reliability table to decide how much weight it deserves. Peers in the same zone may share a source.",
]
HAND_PRIOR = ("Physical priors you may use: leak jets radiate stationary broadband noise mostly between 200 and 1000 Hz that repeats night after night; traffic, pumps, valves and domestic draw are intermittent or low frequency; sound pipes usually concentrate energy below 200 Hz. ")
KEYS_T = "site, perception (max 35 words), temporal (max 35 words), spatial (max 30 words), synthesis (max 50 words), leak_probability (number 0 to 1), verdict (LEAK, NO LEAK, or AMBIGUOUS), action (DISPATCH EXCAVATION CREW, EXTEND MONITORING, or NO EXCAVATION), and work_order (max 50 words, written for a utility engineer)"
KEYS_X = KEYS_T.replace("synthesis (max 50 words)", "ml_and_cases (max 35 words), synthesis (max 50 words)")

def agentic_text(cond):
    roles = ("Run the roles in order for EACH site, judging every site independently from its own tool outputs and the reference tables. "
             "1. Perception agent: interpret spectral evidence (energy bands, centroid, flatness, envelope). "
             "2. Temporal agent: interpret multi-night persistence versus transient disturbance. "
             "3. Spatial agent: interpret the peer-site context in the same zone (a leak is audible only over tens of metres, so a real leak usually lifts neighbouring loggers too, and an isolated signature may reflect a local noise source). Distances to leaks are unknown. ")
    extra = ""
    if cond in ("TML", "H"):
        roles += "4. Model agent: interpret the random forest probability for the site and judge how much weight it deserves from its reliability table. "
        extra += "The random forest was trained only on other zones; its reliability table shows how often its probabilities in each bin were truly leaking on other zones. "
    if cond in ("TRAG", "H"):
        roles += ("%s. Case agent: compare the site with the similar confirmed cases from other zones, noting how closely they match and what their outcomes were. " % ("5" if cond == "H" else "4"))
    last = {"T": "4", "TML": "5", "TRAG": "5", "H": "6"}[cond]
    roles += (f"{last}. Chief agent: synthesise all findings, resolve conflicts, state a calibrated probability that a leak lies within audible range of the logger, and issue a dispatch decision. ")
    txt = "You are the Chief Diagnostic Deliberation Agent in a multi-agent acoustic leak diagnosis system for urban water distribution mains. The input holds deterministic tool outputs for several anonymised loggers (sites) in one zone, plus reference tables built from other zones with confirmed ground truth. " + roles + extra
    if cond in ("T", "TML", "TRAG"):
        txt += HAND_PRIOR
    if cond in ("TRAG", "H"):
        txt += "Background notes (general, not rules): " + " ".join(f"({i+1}) {n}" for i, n in enumerate(NOTES)) + " "
    if cond == "H":
        txt += "The data_derived_band_profile shows which frequency bands separated leaking from sound deployments in the other zones; rely on it and on the similar cases rather than on fixed assumptions about frequency. "
    txt += "Some sites show conflicting evidence, so weigh it honestly and do not default to extreme probabilities. Return ONLY a JSON array (no markdown fence, no commentary), one object per site, with exactly these keys: " + (KEYS_T if cond == "T" else KEYS_X) + "."
    return txt
ZS = ("The input is a table of summary acoustic features, one entry per logger (site), from nocturnal vibration recordings of buried water mains (4,096 Hz sampling, 0 to 2,048 Hz band). For each site, estimate the probability that a leak is present near the logger. "
      "Return ONLY a JSON array (no markdown fence, no commentary), one object per site, with exactly these keys: site, leak_probability (number 0 to 1), verdict (LEAK or NO LEAK), reason (max 30 words).")

def packet(cond, z):
    p = full[z]; out = {"zone": z}
    if cond == "Z":
        out["sites"] = [dict(site=s["site"], **s["_zs"]) for s in p["sites"]]; return out
    out["reference_from_other_zones"] = p["reference_from_other_zones"]
    if cond == "H": out["data_derived_band_profile"] = p["data_derived_band_profile"]
    if cond in ("TML", "H"): out["random_forest_reliability_on_other_zones"] = p["rf_reliability_on_other_zones"]
    sites = []
    for s in p["sites"]:
        o = {k: s[k] for k in ("site", "perception_tool_output", "temporal_tool_output", "spatial_tool_output")}
        if cond in ("TML", "H"): o["ml_tool_output"] = s["ml_tool_output"]
        if cond in ("TRAG", "H"): o["similar_cases_from_other_zones"] = s["similar_cases_from_other_zones"]
        sites.append(o)
    out["sites"] = sites; return out

INSTR = {"Z": ZS, **{c: agentic_text(c) for c in ("T", "TML", "TRAG", "H")}}
json.dump(INSTR, open(os.path.join(V3, "prompts_v3.json"), "w"), indent=1)
for cond in ("Z", "T", "TML", "TRAG", "H"):
    for z in full:
        json.dump(dict(instructions=INSTR[cond], input=packet(cond, z)), open(os.path.join(INP, f"{cond}_{z}.json"), "w"), separators=(",", ":"))
print("files:", len(os.listdir(INP)), "| sizes H_A:", os.path.getsize(os.path.join(INP, "H_A.json")), "T_A:", os.path.getsize(os.path.join(INP, "T_A.json")))
print("\nH prompt:\n", INSTR["H"][:1800])
# leakage self-check: no identifying strings in any input
bad = json.load(open(os.path.join(ROOT, "private", "sites.json")))["name_fragments"] + ["negative", "control", "distance_m", "excavat"]  # private file, not in repo
txt = "".join(open(os.path.join(INP, f), encoding="utf-8").read() for f in os.listdir(INP) if not f.startswith("Z_"))
print("identifier leak check (expected only instruction words):", {b: txt.count(b) for b in bad if txt.count(b)})
