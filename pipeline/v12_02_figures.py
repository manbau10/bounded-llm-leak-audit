"""v11 step 2: all Paper 6 figures as one editable PowerPoint deck (native shapes, text and charts).
Produces manuscript/Paper6_Figures.pptx. Slides are exported to figures/v11/fig1..6.png by v11_03_export.ps1."""
import os, sys, json, datetime, re, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Pt
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE))
from p6figlib import *  # noqa
F = ROOT / "derived" / "v12_final"; FIG = ROOT / "figures" / "v12"; PPT = ROOT / "manuscript" / "Paper6_Figures_revised.pptx"
M = pd.read_csv(F / "main_metrics.csv").set_index("model"); SITE = pd.read_csv(F / "site_level.csv")
TRI = pd.read_csv(F / "main_triage.csv"); AUD = pd.read_csv(F / "main_audit.csv").set_index("model")
EXT = pd.read_csv(F / "external_metrics.csv"); EAU = pd.read_csv(F / "external_audit.csv"); SUM = json.load(open(F / "summary.json"))
LLMS = ["Claude Opus 5.5", "GPT-6 Codex", "Gemini 4 Hydrogen"]
SHORT = {"Claude Opus 5.5": "Opus 5.5", "Claude Sonnet 5.5": "Sonnet 5.5", "GPT-6 Codex": "GPT 6.1 sol", "Gemini 4 Hydrogen": "Gemini 4"}
DISP = {"Claude Opus 5.5": "Claude Opus 5.5", "GPT-6 Codex": "GPT 6.1 sol", "Gemini 4 Hydrogen": "Gemini 4 Hydrogen"}
ZNO = {z: i + 1 for i, z in enumerate("ABCDEFGHIJKLMNO")}
def lab(code): return f"{ZNO[code[0]]}.{code[1:]}"


def new_slide(prs): return prs.slides.add_slide(prs.slide_layouts[6])


# ============================================================== Figure 1: research framework
def fig1(prs):
    s = new_slide(prs)
    G, O, B, V, R = GREEN, ORANGE, BLUE, PURPLE, RED
    # 1 field data
    container(s, 6, 6, 300, 318, "1  Field data", G, "F4FAF6")
    s.shapes.add_picture(str(FIG / "photo_logger.png"), Pt(16), Pt(40), Pt(104), Pt(103)).line.color.rgb = rgb(INK2)
    for i, t in enumerate(["4,096 Hz, 10 s", "15 sites", "70 deployments"]):
        box(s, 130, 40 + i * 36, 166, 31, t, line=G, lw=1.5, bold=(i == 2))
    wave(s, 18, 150, 278, 24, BLUE)
    text(s, 16, 176, 284, 26, "2,790 of 3,188 night recordings kept", T, False, INK2, PP_ALIGN.CENTER)
    box(s, 16, 206, 280, 32, "20 leaking, 50 sound", fill="F1F0EC", line=INK2)
    box(s, 16, 244, 280, 32, "13 in two leak-free control sites", fill="EAF5EC", line=G)
    box(s, 16, 282, 280, 32, "+ Dongguan and Sheffield (public)", fill="FFFFFF", line=INK2)
    block_arrow(s, 309, 150, 18, 26)
    # 2 deterministic evidence tools
    container(s, 330, 6, 300, 318, "2  Deterministic evidence tools", O, "FFF8F2")
    y0 = 42
    rows = [("Spectral", "band shares,\ncentroid, flatness"), ("Temporal", "persistence,\ntransient ratio"),
            ("Spatial", "leak-like peers\nin the same site"), ("Retrieval", "5 labelled cases\nfrom other sites"),
            ("ML tool", "random forest,\ntop 4 effects")]
    for i, (h, t) in enumerate(rows):
        y = y0 + i * 56; x = 340
        if i == 0:
            for k, hh in enumerate([10, 26, 18, 12, 6]): shape(s, MSO_SHAPE.RECTANGLE, x + 2 + k * 8, y + 34 - hh, 6, hh, fill=O)
        elif i == 1: grid(s, x, y + 6, 3, 5, 6, [["E2622B" if (r + c) % 3 else "F6D2BF" for c in range(5)] for r in range(3)], 2)
        elif i == 2:
            for k in range(3): shape(s, MSO_SHAPE.OVAL, x + 2 + k * 14, y + 10 + (k % 2) * 10, 10, 10, fill=O if k == 1 else "F6D2BF", line=O)
        elif i == 3: magnifier(s, x + 4, y + 4, 28, O)
        else:
            shape(s, MSO_SHAPE.ISOSCELES_TRIANGLE, x + 4, y + 4, 34, 26, fill="7DB46C", line=GREEN)
            shape(s, MSO_SHAPE.RECTANGLE, x + 18, y + 30, 6, 8, fill="8B5A2B")
        text(s, x + 46, y - 2, 100, 24, h, T, True, O)
        text(s, x + 140, y - 2, 152, 48, t, T, False, INK)
    block_arrow(s, 633, 150, 18, 26)
    # 3 bounded LLM audit
    container(s, 654, 6, 300, 318, "3  Bounded LLM audit", B, "F3F7FD")
    box(s, 664, 38, 282, 30, "Explained RF score, p_RF", fill="FFFFFF", line=INK2, bold=True)
    arrow(s, 805, 68, 805, 76, INK2)
    for i, m in enumerate(LLMS):
        c = MODEL_COL[m]; xx = 664 + i * 96; yy = 76
        box(s, xx, yy, 90, 50, "GPT 6.1\nsol" if m == "GPT-6 Codex" else SHORT[m], fill="FFFFFF", line=c, color=c, bold=True, size=17)
    box(s, 664, 150, 282, 32, "adjustment a in {−2, −1, 0, +1, +2}", fill="FFFFFF", line=B, color=B, bold=True)
    box(s, 664, 188, 282, 34, "p′ = σ(logit p_RF + 0.7a)", fill="E6EEFA", line=B, size=20, bold=True)
    for i, t in enumerate(["Rationale", "Action", "Work order"]):
        x = 668 + i * 95; doc_icon(s, x + 34, 230, 22, 28, B); text(s, x, 260, 92, 26, t, T, False, INK, PP_ALIGN.CENTER)
    text(s, 664, 290, 282, 26, "One isolated call per model and site", 17, False, INK2, PP_ALIGN.CENTER)
    block_arrow(s, 140, 327, 26, 18, "down")
    # 4 blind evaluation
    container(s, 6, 348, 476, 186, "4  Blind evaluation without site leakage", V, "F7F5FD")
    cols = [["D88A2B" if r == c else "DCD8F2" for c in range(15)] for r in range(15)]
    grid(s, 18, 382, 15, 15, 6.6, cols, 1.6)
    text(s, 12, 508, 150, 20, "15 folds, one site out", 15, False, INK2)
    shape(s, MSO_SHAPE.RECTANGLE, 162, 384, 16, 16, fill="D88A2B"); text(s, 182, 379, 120, 24, "held-out site", T)
    shape(s, MSO_SHAPE.RECTANGLE, 162, 412, 16, 16, fill="DCD8F2"); text(s, 182, 407, 120, 24, "training sites", T)
    padlock(s, 164, 444, 20, R)
    text(s, 190, 438, 130, 70, "No names, labels,\ndistances or\ncontrol flag", T, False, R)
    text(s, 322, 380, 156, 50, "One fresh call\nper held-out site", T, False, INK)
    for k in range(12): shape(s, MSO_SHAPE.OVAL, 330 + (k % 4) * 16, 446 + (k // 4) * 16, 11, 11, fill=(RED if k in (1, 6, 11) else "9A9994"))
    text(s, 322, 498, 156, 34, "4,000 resamples", T, False, INK)
    block_arrow(s, 485, 428, 14, 24)
    # 5 outputs
    container(s, 502, 348, 452, 186, "5  Outputs", R, "FDF6F5")
    text(s, 512, 380, 210, 24, "O1  Discrimination", T, True, INK)
    for k, (hh, c) in enumerate([(46, DARK), (44, MODEL_COL[LLMS[0]]), (44, MODEL_COL[LLMS[2]]), (30, "BDBDBD")]):
        shape(s, MSO_SHAPE.RECTANGLE, 520 + k * 26, 452 - hh, 20, hh, fill=c)
    text(s, 625, 410, 100, 44, "AUC, F1,\nrecall", T, False, INK2)
    text(s, 732, 380, 220, 24, "O2  Audit behaviour", T, True, INK)
    for k, c in enumerate(["EEEEEE"] * 3 + ["6BAED6", "EEEEEE", "FC9272", "EEEEEE", "6BAED6", "EEEEEE", "EEEEEE"]):
        shape(s, MSO_SHAPE.RECTANGLE, 740 + k * 19, 414, 16, 26, fill=c)
    text(s, 512, 462, 210, 24, "O3  Action tiers", T, True, INK)
    for k, (t, c) in enumerate([("D", RED), ("E", "E7A93B"), ("H", "9A9994")]):
        box(s, 520 + k * 40, 492, 34, 32, t, fill=c, line=c, color="FFFFFF", bold=True)
    text(s, 732, 462, 220, 24, "O4  External replication", T, True, INK)
    text(s, 740, 490, 210, 40, "Dongguan, 964 clips\nSheffield, 230 records", 17, False, INK2)
    notes(s, "Figure 1: Research framework")


# ============================================================== Figure 2: anatomy of one audit
def fig2(prs):
    s = new_slide(prs)
    pk = json.load(open(ROOT / "derived/v5/inputs/H2_M.json", encoding="utf-8"))["input"]
    st = [x for x in pk["sites"] if x["site"] == "M6"][0]; r = SITE.set_index("anon").loc["M6"]
    container(s, 6, 6, 360, 528, f"Evidence packet, logger {lab('M6')}", ORANGE, "FFF8F2")
    bands = st["perception_tool_output"]["band_energy_fractions"]
    text(s, 16, 38, 340, 24, "Band energy share (%)", T, True, INK)
    labs = ["0–200", "200–600", "600–1k", "1k–1.5k", "1.5k–2k"]
    for k, (kk, v) in enumerate(bands.items()):
        x = 28 + k * 66; hh = v * 170
        shape(s, MSO_SHAPE.RECTANGLE, x, 160 - hh, 44, hh, fill=ORANGE if 1 <= k <= 2 else "F2B38F")
        text(s, x - 8, 160 - hh - 24, 60, 22, f"{100 * v:.0f}", T, False, INK, PP_ALIGN.CENTER)
        text(s, x - 14, 162, 72, 22, labs[k], 15, False, INK2, PP_ALIGN.CENTER)
    line(s, 20, 160, 356, 160, INK2, 1.0)
    tt = st["temporal_tool_output"]; pp = st["perception_tool_output"]
    rich(s, 16, 188, 344, 80, [[("Persistence ", True, INK), (f"{tt['nightly_persistence_ratio']:.2f} over {tt['n_nights']} nights", False, INK)],
                               [("Flatness ", True, INK), (f"{pp['wiener_spectral_flatness']:.2f}", False, INK), ("   Transient ", True, INK), (f"{tt['transient_ratio_p95_over_median_rms']:.2f}", False, INK)],
                               [("Leak-like peers ", True, INK), (f"{st['spatial_tool_output']['n_peer_sites_with_leak_like_signature']} of {st['spatial_tool_output']['n_peer_sites_in_same_zone']}", False, INK)]])
    text(s, 16, 272, 344, 24, f"Random forest p = {st['random_forest']['leak_probability']:.2f}, largest effects", T, True, INK)
    nm = {"spectral centroid": "Centroid", "transient ratio": "Transient", "nightly persistence": "Persistence", "mean RMS": "Mean RMS"}
    for k, e in enumerate(st["random_forest"]["local_explanation_top_features"]):
        y = 302 + k * 30; w = e["effect_on_probability"] * 900
        text(s, 16, y - 2, 120, 26, nm.get(e["feature"], e["feature"]), T)
        shape(s, MSO_SHAPE.RECTANGLE, 140, y + 3, w, 18, fill=BLUE)
        text(s, 144 + w, y - 2, 70, 26, f"+{e['effect_on_probability']:.2f}", T, False, BLUE)
    text(s, 16, 424, 344, 24, "Five most similar cases, other sites", T, True, INK)
    for k, c in enumerate(st["similar_cases_from_other_zones"]):
        col = LEAK if c["outcome"] == "leaking" else SOUND
        shape(s, MSO_SHAPE.OVAL, 26 + k * 64, 456, 30, 30, fill=col)
        text(s, 6 + k * 64, 488, 70, 22, f"d {c['distance']:.1f}", 15, False, INK2, PP_ALIGN.CENTER)
    text(s, 16, 508, 340, 24, "red: leaking, grey: sound", 15, False, INK2)
    block_arrow(s, 370, 255, 18, 28)
    container(s, 392, 6, 562, 528, "Three independent blind audits of the same packet", BLUE, "F3F7FD")
    AC = {'DISPATCH EXCAVATION CREW': 'dispatch crew', 'EXTEND MONITORING': 'extend monitoring', 'NO EXCAVATION': 'hold excavation'}
    for i, m in enumerate(LLMS):
        c = MODEL_COL[m]; x = 402; y = 40 + i * 152
        shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, 542, 144, fill="FFFFFF", line=c, lw=1.75).adjustments[0] = 0.06
        chip(s, x + 12, y + 12, 20, c); text(s, x + 40, y + 6, 240, 26, DISP[m], T, True, c)
        a = int(r[f"a_{m}"]); pnew = r[f"p_{m}"]
        rich(s, x + 262, y + 6, 278, 26, [[("a = ", False, INK), ((f"{a:+d}" if a else "0"), True, c), (f"   p {st['random_forest']['leak_probability']:.2f} \u2192 {pnew:.2f}", False, INK)]])
        rich(s, x + 12, y + 36, 520, 26, [[("Recommendation: ", False, INK), (AC[r[f'act_{m}']], True, INK)]])
        text(s, x + 12, y + 64, 520, 78, "\u201c" + str(r[f"audit_reason_{m}"]) + "\u201d", 16.5, False, INK2, italic=True)
    text(s, 402, 500, 540, 28, "Field outcome: leaking. All three recommended dispatch.", T, True, GREEN)
    notes(s, "Figure 2: One audit, end to end")


# ============================================================== Figure 3: main benchmark
def fig3(prs):
    s = new_slide(prs)
    names = [m for m in M.index if not m.startswith("LLM own")]
    order = sorted(names, key=lambda m: -M.loc[m, "AUC"])
    def disp(m):
        if m.startswith("Audited RF, "): return "RF audited by " + SHORT[m.split(", ")[1]]
        return {"Random forest, benchmark": "Random forest", "Random forest, audited tool": "RF tool before audit"}.get(m, m)
    def col(m):
        if m.startswith("Audited RF, "): return MODEL_COL[m.split(", ")[1]]
        if m == "Random forest, audited tool": return "7A7A7A"
        if m.startswith("Random forest"): return DARK
        if m in ("Zero-shot LLM",): return RED
        if m in ("Rule engine", "Level-and-spread heuristic"): return "D9A441"
        return "BDBDBD"
    rev = order[::-1]
    text(s, 250, 4, 400, 26, "AUC, 95% CI by site", 20, True, INK)
    text(s, 720, 4, 240, 26, "F1 score, p ≥ 0.5", 20, True, INK)
    ch = bars(s, 6, 30, 640, 506, [disp(m) for m in rev], [M.loc[m, "AUC"] for m in rev], [col(m) for m in rev], True, 0.5, 1.0,
              ([M.loc[m, "AUC_clo"] for m in rev], [M.loc[m, "AUC_chi"] for m in rev]), None, 0.1, "0.0", gap=45, cat_size=17)
    plot_layout(ch, 0.42, 0.01, 0.55, 0.90)
    ch2 = bars(s, 650, 30, 304, 506, [disp(m) for m in rev], [M.loc[m, "F1"] for m in rev], [col(m) for m in rev], True, 0.0, 1.0,
               None, None, 0.25, "0.00", gap=45, labels=True, show_cats=False)
    plot_layout(ch2, 0.02, 0.01, 0.80, 0.90)
    notes(s, "Figure 3: Blind leave-one-site-out benchmark")


# ============================================================== Figure 4: audit behaviour map
def fig4(prs):
    s = new_slide(prs)
    d = SITE.sort_values("p_Random forest (tool used by H2)").reset_index(drop=True); n = len(d)
    x0, cw, gap = 222, 7.6, 1.25; xr = x0 + n * (cw + gap)
    text(s, x0, 4, xr - x0, 26, "70 deployments, sorted by random forest score →", 20, True, INK, PP_ALIGN.CENTER)
    def strip(y, h, colours, label, sub=None, lc=INK):
        text(s, 6, y + h / 2 - 13, 144, 26, label, T, True, lc)
        if sub: text(s, 150, y + h / 2 - 13, 72, 26, sub, 16, False, INK2)
        for i, c in enumerate(colours): shape(s, MSO_SHAPE.RECTANGLE, x0 + i * (cw + gap), y, cw, h, fill=c)
    truth = [CTRL if r.is_negative_control else (LEAK if r.label == 1 else SOUND) for r in d.itertuples()]
    strip(36, 26, truth, "Field truth")
    def grey(p): v = int(235 - 200 * p); return f"{v:02X}{v:02X}{v:02X}"
    rfp = d["p_Random forest (tool used by H2)"].values
    strip(68, 26, [grey(p) for p in rfp], "RF score")
    k = int((rfp < 0.5).sum()); xt = x0 + k * (cw + gap) - gap / 2
    ACOL = {-2: "08306B", -1: "5B9BD5", 0: "EDEDEB", 1: "F4877A", 2: "C0392B"}
    TCOL = {"DISPATCH EXCAVATION CREW": "C0392B", "EXTEND MONITORING": "E7A93B", "NO EXCAVATION": "D3D2CE"}
    y = 104
    for m in LLMS:
        c = MODEL_COL[m]
        strip(y, 22, [ACOL[int(a)] for a in d[f"a_{m}"]], SHORT[m], "audit", c)
        strip(y + 25, 16, [TCOL[a] for a in d[f"act_{m}"]], "", "action")
        a = AUD.loc[m]; text(s, xr + 4, y - 2, 960 - xr - 6, 44, f"{int(a.changed)} changed\n{int(a.broken)} broken", 15, False, INK2)
        y += 52
    line(s, xt, 32, xt, y - 4, INK, 1.5, dash=True); text(s, xt - 150, y - 4, 146, 24, "RF = 0.5", 16, False, INK, PP_ALIGN.RIGHT)
    bi = [i for i, r in enumerate(d.itertuples()) if r.anon == "N2"][0]; xb = x0 + bi * (cw + gap) + cw / 2
    arrow(s, xb + 30, y + 30, xb, y - 4, RED, 1.75)
    text(s, xb + 34, y + 22, 948 - xb - 34, 26, f"Logger {lab('N2')}: lowered by all three", 17, False, RED)
    yl = y + 64; xx = 20
    for t, c in [("Leaking", LEAK), ("Sound", SOUND), ("Leak-free control", CTRL)]:
        shape(s, MSO_SHAPE.RECTANGLE, xx, yl + 4, 16, 16, fill=c); text(s, xx + 20, yl, 170, 24, t, T); xx += 40 + len(t) * 9
    xx = 20; yl += 32
    for v in (-2, -1, 0, 1, 2):
        shape(s, MSO_SHAPE.RECTANGLE, xx, yl + 4, 16, 16, fill=ACOL[v]); text(s, xx + 20, yl, 70, 24, ("a = " + ("−" if v < 0 else "+") + str(abs(v))) if v else "a = 0", T); xx += 98
    xx += 10
    for t, c in [("dispatch", "C0392B"), ("extend", "E7A93B"), ("hold", "D3D2CE")]:
        shape(s, MSO_SHAPE.RECTANGLE, xx, yl + 4, 16, 16, fill=c); text(s, xx + 20, yl, 90, 24, t, T); xx += 104
    ag = SUM["agreement"]
    box(s, 6, yl + 40, 948, 58, f"All three auditors left {ag['all_zero']} of {SUM['n']} scores unchanged and reached the same leak or no-leak decision for all {ag['same_decisions']} deployments.",
        fill="F7F6F2", line="D3D2CE", size=T)
    notes(s, "Figure 4: Where the auditors intervened")


# ============================================================== Figure 5: triage outcomes
def fig5(prs):
    s = new_slide(prs)
    acts = [("Dispatch", "DISPATCH EXCAVATION CREW", "C0392B"), ("Extend", "EXTEND MONITORING", "E7A93B"), ("Hold", "NO EXCAVATION", "8C8B87")]
    text(s, 6, 6, 900, 28, "Rows: recommended action. Columns: field truth.", T, False, INK2)
    gx, tw, th = 104, 66, 62
    for i, m in enumerate(LLMS):
        x = gx + i * 172; c = MODEL_COL[m]
        text(s, x, 40, 160, 26, SHORT[m], 20, True, c, PP_ALIGN.CENTER)
        for j, h in enumerate(["Leak", "Sound"]): text(s, x + 8 + j * (tw + 6), 68, tw, 24, h, 17, False, INK2, PP_ALIGN.CENTER)
        t = TRI[TRI.model == m].set_index("action")
        for r_, (an, ak, ac) in enumerate(acts):
            yy = 94 + r_ * (th + 6)
            for j, col in enumerate(["leaking", "sound"]):
                v = int(t.loc[ak, col]); good = (ak == "DISPATCH EXCAVATION CREW" and j == 0) or (ak == "NO EXCAVATION" and j == 1)
                fill = ac if good else ("FBEFD9" if ak == "EXTEND MONITORING" else "EEEDE9")
                box(s, x + 8 + j * (tw + 6), yy, tw, th, str(v), fill=fill, line=fill, color=("FFFFFF" if good else INK), size=22, bold=True)
        if i == 0:
            for r_, (an, _, ac) in enumerate(acts): text(s, 6, 94 + r_ * (th + 6) + 18, 96, 26, an, T, True, ac)
        d_ = t.loc["DISPATCH EXCAVATION CREW"]; h_ = t.loc["NO EXCAVATION"]
        text(s, x, 302, 160, 52, f"Hold right\n{int(h_.sound)}/{int(h_.n)} ({100 * h_.sound / h_.n:.0f}%)", 17, False, INK, PP_ALIGN.CENTER)
        text(s, x, 352, 160, 52, f"Dispatch right\n{int(d_.leaking)}/{int(d_.n)} ({100 * d_.leaking / d_.n:.0f}%)", 17, False, INK, PP_ALIGN.CENTER)
    # random forest alone
    x = gx + len(LLMS) * 172 + 8; rf = SITE["p_Random forest (tool used by H2)"].values; y = SITE.label.values
    container(s, x - 6, 36, 960 - x, 330, "", INK2)
    text(s, x, 40, 160, 26, "RF alone", 20, True, DARK, PP_ALIGN.CENTER)
    for j, h in enumerate(["Leak", "Sound"]): text(s, x + 8 + j * (tw + 6), 68, tw, 24, h, 17, False, INK2, PP_ALIGN.CENTER)
    for r_, (an, pos) in enumerate([("p ≥ 0.5", True), ("p < 0.5", False)]):
        yy = 94 + (0 if pos else 2) * (th + 6)
        for j, lv in enumerate([1, 0]):
            v = int(((rf >= .5) == pos)[y == lv].sum()); good = (pos and lv == 1) or (not pos and lv == 0)
            box(s, x + 8 + j * (tw + 6), yy, tw, th, str(v), fill=(DARK if good else "EEEDE9"), line=(DARK if good else "EEEDE9"), color=("FFFFFF" if good else INK), size=22, bold=True)
        text(s, x + 8, yy + th + 2 if pos else yy - 26, 140, 24, an, 16, False, INK2, PP_ALIGN.CENTER)
    text(s, x - 4, 302, 160, 52, "No middle tier;\nevery site decided", 17, False, INK, PP_ALIGN.CENTER)
    # bottom message strip
    ext = TRI[TRI.action == "EXTEND MONITORING"]; hold = TRI[TRI.action == "NO EXCAVATION"]; hr = 100 * hold.sound / hold.n
    box(s, 6, 420, 948, 108, "Extended monitoring absorbed {0} to {1} deployments, of which {2} to {3} were leaking.\nHold decisions were correct for {4:.0f}% to {5:.0f}% of deployments across the three auditors.".format(
        int(ext.n.min()), int(ext.n.max()), int(ext.leaking.min()), int(ext.leaking.max()), hr.min(), hr.max()), fill="F7F6F2", line="D3D2CE", size=T)
    notes(s, "Figure 5: Three-tier triage against field truth")


# ============================================================== Figure 6: external replication
def fig6(prs):
    s = new_slide(prs)
    cfg = [("Dongguan", 6, f"{SUM['external']['dongguan']['n']} clips, {SUM['external']['dongguan']['groups']} groups, 5 folds", 0.90, 0.98, 0.02),
           ("Sheffield", 486, f"{SUM['external']['sheffield']['n']} records, 6 groups, 3 folds", 0.70, 1.00, 0.05)]
    order = ["Random forest"] + [f"Audited RF, {m}" for m in LLMS]
    cols = [DARK] + [MODEL_COL[m] for m in LLMS]
    for ds, x, sub, lo, hi, unit in cfg:
        container(s, x, 6, 468, 528, ds, GREEN if ds == "Dongguan" else PURPLE, "FFFFFF")
        text(s, x + 120, 10, 340, 24, sub, T, False, INK2)
        e = EXT[EXT.dataset == ds].set_index("model")
        ch = bars(s, x + 6, 40, 456, 280, order, [e.loc[m, "AUC"] for m in order], cols, False, lo, hi,
                  ([e.loc[m, "AUC_cluster_lo"] for m in order], [e.loc[m, "AUC_cluster_hi"] for m in order]), "AUC", unit, "0.00", gap=50, show_cats=False)
        px, pw = 0.17, 0.80; plot_layout(ch, px, 0.04, pw, 0.86)
        nb = len(order); cx = [x + 6 + 456 * (px + pw * (k + 0.5) / nb) for k in range(nb)]
        names = ["RF", "Opus\n5.5", "GPT 6.1\nsol", "Gemini\n4"]
        for k in range(nb): text(s, cx[k] - 40, 322, 80, 46, names[k], 17, True, cols[k], PP_ALIGN.CENTER)
        a = EAU[EAU.dataset == ds].set_index("model")
        rowsl = [("Changed", "adjusted"), ("Fixed", "fixed"), ("Broken", "broken")]
        for r_, (rl, key) in enumerate(rowsl):
            yy = 374 + r_ * 34
            text(s, x + 8, yy, 80, 28, rl, 17, True, INK)
            for k in range(nb):
                v = "–" if k == 0 else str(int(a.loc[LLMS[k - 1], key]))
                fill = {"fixed": "DFF0E3", "broken": "F8DEDA"}.get(key, "F1F0EC")
                box(s, cx[k] - 26, yy, 52, 28, v, fill=fill, line=fill, size=17, bold=True)
        acc = [e.loc[m, "accuracy"] for m in order]
        text(s, x + 8, 478, 452, 50, "Accuracy " + ", ".join(f"{100 * v:.1f}" for v in acc) + " %", 16, False, INK2)
    notes(s, "Figure 6: External replication on two public datasets")


def finalize(path):
    tmp = str(path) + ".tmp"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for it in zin.infolist():
            data = zin.read(it.filename)
            if it.filename == "docProps/app.xml":
                data = re.sub(rb"<Application>[^<]*</Application>", b"<Application>Microsoft Office PowerPoint</Application>", data)
            zout.writestr(it, data)
    os.replace(tmp, path)


if __name__ == "__main__":
    prs = Presentation(); prs.slide_width, prs.slide_height = Pt(960), Pt(540)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6): f(prs)
    cp = prs.core_properties; cp.author = cp.last_modified_by = "Ridwan Taiwo"; cp.title = "Paper 6 figures"; cp.comments = cp.subject = cp.keywords = ""
    now = datetime.datetime(2026, 11, 24, 12, 0, 0); cp.created = cp.modified = now; cp.revision = 1
    prs.save(PPT); finalize(PPT); print("saved", PPT, len(prs.slides))
