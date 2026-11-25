"""v10: Claude Opus 5.5 arm on the frozen v7 (Dongguan + Sheffield transfer) and v8 (Sheffield-local) packets.
Prompts are byte-identical to v7/v8 (prompt_sha256 matches the Sonnet responses). Opus ran as isolated Claude Code
subagents (one Read of its packet, one Write of its answer), because Opus via Poe returned 404 earlier.
Usage: python v10_opus_external.py validate SCRATCH_DIR   |   python v10_opus_external.py score SCRATCH_DIR
Scoring reuses v7 metrics() and bootstrap() unchanged. Earlier v7/v8 outputs are read only."""
import sys, json, glob, os, hashlib, shutil
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import v7_external_hybrids as base
V7 = os.path.join(ROOT, "derived", "v7_external_hybrids"); V8 = os.path.join(ROOT, "derived", "v8_sheffield_local")
OUT = os.path.join(ROOT, "derived", "v10_opus_external"); M = "claude-opus-5.5"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

def validate(S):
    man = json.load(open(os.path.join(S, "manifest.json"))); ok, bad = [], []
    for n in sorted(man):
        tag, pk = n.split("_"); pdir = V7 if tag == "v7" else V8
        want = [t["site"] for t in json.load(open(os.path.join(pdir, "packets", pk + ".json")))["targets"]]
        f = os.path.join(S, "out", n + ".json")
        try:
            assert base.textsha(open(os.path.join(S, "in", n + ".txt"), encoding="utf-8").read()) == man[n], "prompt changed"
            base.parse(open(f, encoding="utf-8").read(), want); ok.append(n)
        except Exception as e: bad.append((n, str(e)[:120]))
    print(f"valid {len(ok)}/{len(man)}"); [print(" ", b) for b in bad]; return ok, bad

def score(S):
    ok, bad = validate(S); assert not bad, "incomplete or invalid; do not score"
    man = json.load(open(os.path.join(S, "manifest.json"))); os.makedirs(os.path.join(OUT, "responses"), exist_ok=True)
    for n in ok:
        dst = os.path.join(OUT, "responses", n + ".json")
        if not os.path.exists(dst):
            rows = json.load(open(os.path.join(S, "out", n + ".json"), encoding="utf-8"))["results"]
            base.save(Path_(dst), dict(model=M, packet=n, prompt_sha256=man[n], provider="Claude Code subagent (Read+Write only)", results=rows))
    lockf = os.path.join(OUT, "locked_responses.json")
    hashes = {os.path.basename(p): sha(p) for p in sorted(glob.glob(os.path.join(OUT, "responses", "*.json")))}
    if os.path.exists(lockf): assert json.load(open(lockf)) == hashes, "locked responses changed"
    else: json.dump(hashes, open(lockf, "w"), indent=1)
    rec = {}
    for p in glob.glob(os.path.join(OUT, "responses", "*.json")):
        j = json.load(open(p)); tag = os.path.basename(p)[:2]
        for r in j["results"]: rec[(tag, r["site"])] = r
    others = ["codex-gpt-6.1-sol", "claude-sonnet-5.5", "gemini-4-hydrogen"]
    sets = []
    d7 = pd.read_csv(os.path.join(V7, "scored_predictions.csv"))
    for ds, df in d7.groupby("dataset"): sets.append((f"{ds} (v7 {'transfer' if ds == 'sheffield' else 'grouped CV'})", "v7", df.reset_index(drop=True), "rf_probability"))
    sets.append(("sheffield (v8 local grouped CV)", "v8", pd.read_csv(os.path.join(V8, "local_scored_predictions.csv")), "local_rf"))
    res, pairs, beh = [], [], []
    for name, tag, df, rfc in sets:
        y = df.label.to_numpy(); raw = df[rfc].to_numpy(); q = np.clip(raw, .02, .98)
        a = df.site.map(lambda s: rec[(tag, s)]["audit_adjustment"]).to_numpy(); own = df.site.map(lambda s: float(rec[(tag, s)]["leak_probability"])).to_numpy()
        df["audit_" + M] = a; df["own_" + M] = own; df["reason_" + M] = df.site.map(lambda s: rec[(tag, s)]["audit_reason"])
        df["primary_" + M] = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + .7 * a)))
        probs = {"RF tool": raw}
        for m in [M] + others:
            probs[m + " | primary"] = df["primary_" + m].to_numpy(); probs[m + " | own"] = df["own_" + m].to_numpy(); probs[m + " | mean"] = (raw + df["own_" + m].to_numpy()) / 2
        boot = base.bootstrap(y, probs, df.group.to_numpy())
        for k, p in probs.items():
            lo, hi = np.percentile(boot[k], [2.5, 97.5]); res.append(dict(dataset=name, model=k, n=len(y), AUC_cluster_lo=lo, AUC_cluster_hi=hi, **base.metrics(y, p)))
        for ref in ["RF tool", "claude-sonnet-5.5 | primary"]:
            k = M + " | primary"; dif = boot[k] - boot[ref]; A = (probs[k] >= .5) == y; B = (probs[ref] >= .5) == y; w, l = int((A & ~B).sum()), int((~A & B).sum())
            pairs.append(dict(dataset=name, model=k, reference=ref, delta_AUC=roc_auc_score(y, probs[k]) - roc_auc_score(y, probs[ref]), lo=np.percentile(dif, 2.5), hi=np.percentile(dif, 97.5), fixed=w, broken=l, McNemar=binomtest(w, w + l, .5).pvalue if w + l else 1.))
        for m in [M] + others:
            aa = df["audit_" + m].to_numpy(); oc = (raw >= .5) == y; nc = (df["primary_" + m].to_numpy() >= .5) == y
            beh.append(dict(dataset=name, model=m, adjusted=int((aa != 0).sum()), downgraded=int((aa < 0).sum()), downgraded_true_leaks=int(((aa < 0) & (y == 1)).sum()), upgraded=int((aa > 0).sum()), upgraded_true_leaks=int(((aa > 0) & (y == 1)).sum()), fixed=int((~oc & nc).sum()), broken=int((oc & ~nc).sum())))
        df[["site", "label", "group", rfc, "audit_" + M, "own_" + M, "primary_" + M, "reason_" + M]].to_csv(os.path.join(OUT, f"opus_scored_{tag}_{name.split()[0]}.csv"), index=False)
    pd.DataFrame(res).to_csv(os.path.join(OUT, "metrics.csv"), index=False); pd.DataFrame(pairs).to_csv(os.path.join(OUT, "paired.csv"), index=False); pd.DataFrame(beh).to_csv(os.path.join(OUT, "audit_behavior.csv"), index=False)
    pd.set_option("display.width", 250); r = pd.DataFrame(res); r = r[r.model.str.contains("primary") | (r.model == "RF tool")]
    print(r[["dataset", "model", "AUC", "AUC_cluster_lo", "AUC_cluster_hi", "accuracy", "precision", "sensitivity", "specificity", "F1", "FP", "FN"]].round(3).to_string(index=False))
    print(pd.DataFrame(pairs).round(3).to_string(index=False)); print(pd.DataFrame(beh).to_string(index=False))

from pathlib import Path as Path_
if __name__ == "__main__": {"validate": validate, "score": score}[sys.argv[1]](sys.argv[2])
