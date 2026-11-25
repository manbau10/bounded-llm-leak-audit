"""Score locked H2 outputs without refitting models or changing predictions.

Reuses frozen v5 baseline predictions. Reports original IID bootstrap plus
district-cluster bootstrap sensitivity; all comparisons are post hoc.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib, html, json, platform
import numpy as np
import pandas as pd
from scipy.stats import rankdata, binomtest
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BLIND=ROOT.parent/'Paper6_ChatGPT_blind_run'
V5=ROOT/'derived/v5'
OUT=ROOT/'derived/v6_blind_comparison'
SPECS={'ChatGPT (GPT-6-based Codex)':(BLIND/'outputs','chatgpt'),
       'Claude Sonnet 5.5':(V5/'llm_raw','claude-sonnet'),
       'Gemini 4 Hydrogen':(V5/'llm_raw','gemini-4-hydrogen')}
RF='Random forest (tool used by H2)'
BASE='Random forest'
B=4000

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def sigmoid(x): return 1/(1+np.exp(-x))
def primary(p,a):
    q=np.clip(p,.02,.98)
    return sigmoid(np.log(q/(1-q))+.7*a)

def measures(y,p,ctrl):
    pred=p>=.5
    tp=int(((y==1)&pred).sum()); fn=int(((y==1)&~pred).sum())
    fp=int(((y==0)&pred).sum()); tn=int(((y==0)&~pred).sum())
    return dict(AUC=float(roc_auc_score(y,p)),accuracy=(tp+tn)/len(y),
        balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,
        precision=tp/max(1,tp+fp),recall=tp/(tp+fn),specificity=tn/(tn+fp),
        F1=2*tp/max(1,2*tp+fp+fn),Brier=float(brier_score_loss(y,p)),
        TP=tp,TN=tn,FP=fp,FN=fn,control_FA=int((pred&ctrl).sum()))

def bootstrap_indices(y,districts):
    rng=np.random.default_rng(2026)
    iid=rng.integers(0,len(y),(B,len(y)))
    iid=[i for i in iid if len(np.unique(y[i]))==2]
    rng=np.random.default_rng(2026)
    ds=np.unique(districts); parts={g:np.flatnonzero(districts==g) for g in ds}
    cluster=[]
    for _ in range(B):
        idx=np.concatenate([parts[g] for g in rng.choice(ds,len(ds),replace=True)])
        if len(np.unique(y[idx]))==2: cluster.append(idx)
    return dict(iid=iid,district_cluster=cluster)

def bauc(y,p,indices):
    # Rank statistic is exactly sklearn AUC, including ties.
    vals=[]
    for idx in indices:
        yi=y[idx]; pos=yi==1; n1=pos.sum(); n0=len(idx)-n1
        vals.append((rankdata(p[idx])[pos].sum()-n1*(n1+1)/2)/(n1*n0))
    return np.array(vals)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source_files=[ROOT/'derived/v3/deployments_v3.csv', V5/'predictions_v5.csv',V5/'benchmark_v5.csv',V5/'rf_tool_probabilities.json',ROOT/'PROTOCOL_v3.md',BLIND/'validation_checks.json',BLIND/'external_validation/results.json']
    checks=load(BLIND/'validation_checks.json')
    assert len(checks)==15 and all(c['valid'] for c in checks)
    for c in checks:
        assert digest(BLIND/'outputs'/f"chatgpt_H2_{c['zone']}.json")==c['sha256'], 'Frozen blind prediction changed before scoring'
    for z in 'ABCDEFGHIJKLMNO':
        a=BLIND/'inputs'/f'H2_{z}.json'; b=V5/'inputs'/f'H2_{z}.json'
        assert load(a)==load(b),f'Input packet mismatch in zone {z}'
        source_files += [a,b]
    records={}
    for lab,(folder,prefix) in SPECS.items():
        all_rows=[]
        for z in 'ABCDEFGHIJKLMNO':
            p=folder/f'{prefix}_H2_{z}.json'; source_files.append(p)
            obj=load(p); rows=obj['results'] if isinstance(obj,dict) else obj
            expected={s['site'] for s in load(V5/'inputs'/f'H2_{z}.json')['input']['sites']}
            assert len(rows)==len(expected) and {r['site'] for r in rows}==expected
            all_rows+=rows
        assert len(all_rows)==70 and len({r['site'] for r in all_rows})==70
        for r in all_rows:
            assert type(r['audit_adjustment']) is int and r['audit_adjustment'] in [-2,-1,0,1,2]
            assert 0<=r['leak_probability']<=1
        records[lab]={r['site']:r for r in all_rows}
    before={str(p):digest(p) for p in source_files}
    d=pd.read_csv(ROOT/'derived/v3/deployments_v3.csv')
    cached=pd.read_csv(V5/'predictions_v5.csv').set_index('anon').loc[d.anon].reset_index()
    assert d.anon.is_unique and len(d)==70 and int(d.label.sum())==20 and d.district.nunique()==15
    assert np.array_equal(d.label,cached.label) and np.array_equal(d.is_negative_control,cached.is_negative_control)
    y=d.label.to_numpy(); ctrl=d.is_negative_control.to_numpy().astype(bool)
    probs={c[2:]:cached[c].to_numpy() for c in cached.columns if c.startswith('p_')}
    rfp=d.anon.map(load(V5/'rf_tool_probabilities.json')).to_numpy()
    assert np.allclose(probs[RF],rfp,rtol=0,atol=1e-15)
    audit_rows=[]; originals={}
    for lab,recs in records.items():
        assert set(recs)==set(d.anon)
        arr=[recs[s] for s in d.anon]
        a=np.array([r['audit_adjustment'] for r in arr]); own=np.array([r['leak_probability'] for r in arr])
        originals[lab]=(a,own,arr)
        new={'primary: audited RF':primary(rfp,a),'secondary: mean of RF and LLM':(rfp+own)/2,'LLM own probability':own}
        for method,p in new.items():
            name=f'H2 {lab} ({method})'
            if name in probs: assert np.allclose(p,probs[name],rtol=0,atol=1e-14),f'Existing result mismatch {name}'
            probs[name]=p
        audited=new['primary: audited RF']; changed=a!=0
        better=np.abs(audited-y)<np.abs(rfp-y)-1e-12
        worse=np.abs(audited-y)>np.abs(rfp-y)+1e-12
        old_correct=(rfp>=.5)==y; new_correct=(audited>=.5)==y
        counts={str(v):int((a==v).sum()) for v in [-2,-1,0,1,2]}
        audit_rows.append(dict(model=lab,adjustments=json.dumps(counts),changed=int(changed.sum()),downgraded=int((a<0).sum()),downgraded_true_leaks=int(((a<0)&(y==1)).sum()),upgraded=int((a>0).sum()),upgraded_true_leaks=int(((a>0)&(y==1)).sum()),changed_toward_truth=int((changed&better).sum()),changed_away_from_truth=int((changed&worse).sum()),classification_fixed=int((~old_correct&new_correct).sum()),classification_broken=int((old_correct&~new_correct).sum())))
    indices=bootstrap_indices(y,d.district.to_numpy())
    boot={method:{name:bauc(y,p,idx) for name,p in probs.items()} for method,idx in indices.items()}
    rows=[]
    for name,p in probs.items():
        row=dict(model=name,**measures(y,p,ctrl))
        for method in indices:
            lo,hi=np.percentile(boot[method][name],[2.5,97.5]); row[f'AUC_{method}_lo']=lo; row[f'AUC_{method}_hi']=hi
        rows.append(row)
    metrics=pd.DataFrame(rows).sort_values('AUC',ascending=False)
    # Frozen baseline metrics must reproduce the original report exactly.
    old=pd.read_csv(V5/'benchmark_v5.csv').set_index('model')
    for name in old.index:
        row=metrics.set_index('model').loc[name]
        for col in ['AUC','accuracy','precision','recall','specificity','FP','FN','control_FA']:
            assert abs(row[col]-old.loc[name,col])<1e-12,(name,col)
    metrics.to_csv(OUT/'model_metrics.csv',index=False)
    comparisons=[]
    primaries=[f'H2 {lab} (primary: audited RF)' for lab in SPECS]
    pairs=[(p,ref) for p in primaries for ref in [BASE,RF]]
    pairs += [(primaries[0],p) for p in primaries[1:]]
    for name,ref in pairs:
        a=(probs[name]>=.5)==y; b=(probs[ref]>=.5)==y
        right_only=int((a&~b).sum()); wrong_only=int((~a&b).sum())
        row=dict(model=name,reference=ref,delta_AUC=float(roc_auc_score(y,probs[name])-roc_auc_score(y,probs[ref])),correct_only_model=right_only,correct_only_reference=wrong_only,McNemar_exact_two_sided=float(binomtest(right_only,right_only+wrong_only,.5).pvalue) if right_only+wrong_only else 1.)
        for method in indices:
            diff=boot[method][name]-boot[method][ref]
            lo,hi=np.percentile(diff,[2.5,97.5]); row[f'delta_AUC_{method}_lo']=lo; row[f'delta_AUC_{method}_hi']=hi
            row[f'bootstrap_fraction_delta_le_zero_{method}']=float((diff<=0).mean())
        comparisons.append(row)
    paired=pd.DataFrame(comparisons); paired.to_csv(OUT/'paired_comparisons.csv',index=False)
    audits=pd.DataFrame(audit_rows); audits.to_csv(OUT/'audit_behavior.csv',index=False)
    triage=[]; detailed=d[['anon','zone','label','is_negative_control']].copy()
    for name,p in probs.items(): detailed['p_'+name]=p
    for lab,(a,own,arr) in originals.items():
        detailed['adjustment_'+lab]=a; detailed['action_'+lab]=[r['action'] for r in arr]
        detailed['verdict_'+lab]=[r['verdict'] for r in arr]
        for action in ['DISPATCH EXCAVATION CREW','EXTEND MONITORING','NO EXCAVATION']:
            mask=np.array([r['action']==action for r in arr]); n=int(mask.sum()); leaking=int(y[mask].sum())
            triage.append(dict(model=lab,action=action,n=n,leaking=leaking,sound=n-leaking,negative_controls=int((mask&ctrl).sum())))
    detailed.to_csv(OUT/'site_scores.csv',index=False)
    triage=pd.DataFrame(triage); triage.to_csv(OUT/'triage_outcomes.csv',index=False)
    errors=detailed[['anon','zone','label','is_negative_control']].copy()
    for name in [BASE,RF]+primaries:
        errors[name]=np.where(probs[name]>=.5, np.where(y==1,'TP','FP'),np.where(y==1,'FN','TN'))
    errors=errors[(errors.iloc[:,4:].isin(['FP','FN'])).any(axis=1)]
    errors.to_csv(OUT/'error_cases.csv',index=False)
    save('scoring_provenance.json',dict(date='2026-11-24',scoring_code_sha256=digest(Path(__file__)),n=70,leaking=20,sound=50,districts=15,negative_controls=13,threshold=.5,primary_formula='sigmoid(logit(clip(p_RF_tool,0.02,0.98))+0.7*a)',bootstrap_replicates_requested=B,bootstrap_valid={k:len(v) for k,v in indices.items()},seed=2026,all_existing_v5_metrics_reproduced=True,all_15_blind_input_packets_match=True,models=list(SPECS),chatgpt_identity='GPT-6-based Codex, exact backend revision and decoding settings unavailable',external_llm_evaluations_present=False,source_sha256=before))
    figure(metrics,paired,audits,primaries)
    report(metrics,paired,audits,triage,primaries,indices)
    assert all(digest(Path(p))==h for p,h in before.items()),'A source file changed during scoring'
    for c in checks: assert digest(BLIND/'outputs'/f"chatgpt_H2_{c['zone']}.json")==c['sha256']
    save('verification.json',dict(original_blind_output_hashes_unchanged='15/15',all_source_hashes_unchanged=True,source_files_checked=len(before),existing_v5_metrics_reproduced=True,all_predictions_finite=all(np.isfinite(p).all() for p in probs.values()),no_refitting=True,no_inference_calls=True,no_prediction_files_copied=True))
    print(metrics[metrics.model.isin([BASE,RF]+primaries)][['model','AUC','accuracy','recall','specificity','FP','FN','control_FA']].to_string(index=False))
    print('\nAUDIT BEHAVIOR\n'+audits.to_string(index=False))
    print('\nPAIRED\n'+paired.to_string(index=False))

def short(name):
    return name.replace('H2 ','').replace(' (primary: audited RF)','').replace(' (GPT-6-based Codex)','').replace('Random forest (tool used by H2)','RF tool').replace('Random forest','RF baseline')

def figure(metrics,paired,audits,primaries):
    sel=metrics.set_index('model').loc[[BASE,RF]+primaries].iloc[::-1]
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    z=np.arange(len(sel)); vals=sel.AUC.to_numpy()
    axes[0].errorbar(vals,z,xerr=np.array([vals-sel.AUC_district_cluster_lo,sel.AUC_district_cluster_hi-vals]),fmt='o',capsize=4,color='#236b8c')
    axes[0].set(yticks=z,yticklabels=[short(x) for x in sel.index],xlim=(.6,1.01),xlabel='AUC; 95% district-cluster bootstrap',title='Held-out discrimination')
    for i,(name,r) in enumerate(sel.iterrows()): axes[0].annotate(f'{r.AUC:.3f}',(r.AUC,i),xytext=(5,7),textcoords='offset points',fontsize=9)
    x=np.arange(len(sel)); axes[1].bar(x-.18,sel.FP,.36,label='False positives',color='#d78037'); axes[1].bar(x+.18,sel.FN,.36,label='Missed leaks',color='#427a9d')
    axes[1].set(xticks=x,xticklabels=[short(n) for n in sel.index],ylabel='Deployments',title='Errors at fixed threshold 0.5'); axes[1].tick_params(axis='x',labelrotation=65); axes[1].legend(fontsize=8)
    x=np.arange(len(audits)); axes[2].bar(x-.18,audits.changed_toward_truth,.36,label='Toward truth',color='#38967a'); axes[2].bar(x+.18,audits.changed_away_from_truth,.36,label='Away from truth',color='#bd5a51')
    axes[2].set(xticks=x,xticklabels=[short(n) for n in audits.model],ylabel='Adjusted deployments',title='Direction of probability adjustments'); axes[2].tick_params(axis='x',labelrotation=65); axes[2].legend(fontsize=8)
    fig.savefig(OUT/'model_comparison.png',dpi=180); plt.close(fig)

def report(metrics,paired,audits,triage,primaries,indices):
    def table(df): return df.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}',escape=True)
    main=metrics.set_index('model').loc[[BASE,RF]+primaries].reset_index()
    top=main[['model','AUC','accuracy','precision','recall','specificity','F1','Brier','FP','FN','control_FA']].copy()
    intervals=main[['model','AUC','AUC_iid_lo','AUC_iid_hi','AUC_district_cluster_lo','AUC_district_cluster_hi']]
    chat=main[main.model==primaries[0]].iloc[0]
    ext=load(BLIND/'external_validation/results.json')
    e1=ext['dongguan']['overall']; e2=ext['sheffield_zero_shot']['overall']
    pp=paired[paired.model==primaries[0]][['reference','delta_AUC','delta_AUC_iid_lo','delta_AUC_iid_hi','delta_AUC_district_cluster_lo','delta_AUC_district_cluster_hi','McNemar_exact_two_sided']]
    secondary=metrics[metrics.model.str.startswith('H2 ')][['model','AUC','accuracy','recall','FP','FN','Brier']]
    rows=[]
    for _,a in audits.iterrows():
        rows.append(f"<li><strong>{html.escape(short(a.model))}</strong>: adjusted {a.changed}/70 scores ({a.downgraded} down, {a.upgraded} up). {a.downgraded_true_leaks}/{a.downgraded} downgraded sites were true leaks. Among nonzero adjustments, {a.changed_toward_truth} moved probability toward the observed label and {a.changed_away_from_truth} moved away. Relative to the RF tool, {a.classification_fixed} classification errors were fixed and {a.classification_broken} correct classifications were broken.</li>")
    external_rows=[dict(model='Acoustic RF (separate external baseline)',Dongguan=f"AUC {e1['auc']:.3f}; balanced accuracy {100*e1['balanced_accuracy']:.1f}%",Sheffield=f"AUC {e2['auc']:.3f}; balanced accuracy {100*e2['balanced_accuracy']:.1f}%")]
    external_rows += [dict(model=lab,Dongguan='Not evaluated',Sheffield='Not evaluated') for lab in SPECS]
    pd.DataFrame(external_rows).to_csv(OUT/'external_evaluation_status.csv',index=False)
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Paper 6 — scored blind model comparison</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#1e2c36;max-width:1400px;margin:36px auto;padding:0 25px}}h1,h2{{color:#175674;line-height:1.2}}table{{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto;max-width:100%}}td,th{{padding:9px;border-bottom:1px solid #d0dbe1;text-align:left}}th{{background:#ecf3f6}}.lead{{background:#edf5fa;border-left:5px solid #276e94;padding:18px}}img{{max-width:100%}}code{{background:#eef2f5;padding:2px 4px}}a{{color:#17688c}}</style>
<h1>Paper 6: held-out scoring of the three blind H2 models</h1><p>24 November 2026, Asia/Hong_Kong. Labels were unsealed for scoring after completion of the 15-zone ChatGPT run.</p>
<p class="lead"><strong>ChatGPT H2: AUC {chat.AUC:.4f}, accuracy {100*chat.accuracy:.1f}%, recall {100*chat.recall:.1f}%, {int(chat.FP)} false positives and {int(chat.FN)} missed leaks.</strong> The table below compares all three locked H2 runs under the same scoring rule. The external Dongguan/Sheffield results are from a separate acoustic RF baseline; no ChatGPT, Claude or Gemini external-dataset predictions exist in this work.</p>
<h2>Matched evaluation</h2><p>All three models have exactly the same 70 deployment identifiers across 15 held-out districts. The 15 ChatGPT input packets match the original H2 packets semantically. There are 20 majority-label leaking deployments and 50 sound deployments, including 13 project-designated negative controls. These are the existing study labels, not newly adjudicated ground truth; the project note says the control designation still requires confirmation. Metrics are deployment-level rather than recording-level.</p>
<p>The exact frozen primary score is <code>sigmoid(logit(clip(p_RF_tool, 0.02, 0.98)) + 0.7 × audit_adjustment)</code>; a probability of at least 0.5 is positive. Secondary scores are the mean of the RF-tool probability and the model's own probability, and the own probability alone. Categorical verdicts and actions are analyzed separately, not substituted for the primary score.</p>
<p>The full-precision RF probabilities and all conventional baseline predictions were read from frozen v5 outputs; no model was retrained. Every existing v5 AUC and threshold metric was reproduced. The RF baseline (AUC 0.9025) used standardized features, while the RF tool supplied to H2 (AUC 0.8965) used unscaled features. The original code records different tree tie-breaks. Both are reported: tool-relative comparisons isolate the incremental audit behavior, while baseline-relative comparisons preserve the published protocol.</p>
<h2>Primary comparison</h2>{table(top)}
<p><strong>Interpretation:</strong> all three H2 models give exactly the same 70 binary decisions: 15 true positives, 47 true negatives, three false positives and five false negatives. Claude's AUC is only 0.0005 above ChatGPT's; ChatGPT's is 0.0055 above Gemini's. Both paired district-cluster intervals include zero, so this run does not establish superiority among them. Every H2 model has lower point-estimate AUC and accuracy than both the RF tool and the standardized RF baseline.</p>
<p>Accuracy, precision, recall, specificity and F1 above are fractions. Control_FA counts positive primary scores among the 13 designated controls. Probability calibration was not fitted; Brier score assesses the saved probabilities against the labels.</p>
<img src="derived/v6_blind_comparison/model_comparison.png" alt="Model AUC intervals, errors and directions of audit adjustments">
<h2>Uncertainty and paired comparisons</h2>{table(intervals)}
<p>To match the existing analysis, deployment-IID percentile AUC intervals use 4,000 paired bootstrap resamples, seed 2026. A second analysis resamples whole districts ({len(indices['district_cluster'])} valid two-class draws of 4,000); this addresses within-district dependence more directly. Neither procedure refits the underlying RF or repeats LLM inference, so intervals do not capture all training or generation variability. Only 15 districts are available.</p>
<p>ChatGPT-minus-reference AUC differences and paired intervals:</p>{table(pp)}
<p>McNemar uses the exact two-sided binomial test on discordant classification outcomes. All cross-model comparisons are exploratory, unadjusted for multiplicity. The CSV also includes the legacy one-sided bootstrap fraction of draws with a nonpositive AUC difference; this is reported as a bootstrap fraction, not an exact hypothesis-test p-value. The H2 design was developed after earlier results on these same 70 deployments, and ChatGPT is a post hoc replication. These are not independent confirmations or a revised preregistered primary hypothesis.</p>
<h2>How the audits behaved</h2><ul>{''.join(rows)}</ul>{table(audits)}
<p>The shared additional missed leak was <strong>N2</strong>. Its RF-tool probability was 0.61. ChatGPT and Claude each applied −1, reducing the primary score to 0.437; Gemini applied −2, reducing it to 0.278. ChatGPT's saved rationale emphasized a narrow tonal spectrum and zero persistence over 17 nights. The observed deployment label is leaking. This case illustrates how plausible acoustic counterarguments can suppress a true positive. ChatGPT and Claude deferred N2 for monitoring, while Gemini recommended no excavation. The categorical recommendations differ even though all three primary scores classify it negative.</p>
<p>“Toward truth” means a nonzero adjustment reduced absolute error from the binary observed label; it does not imply better calibration in a new population. Clipping alone can change the RF score slightly when the adjustment is zero, so these direction counts are restricted to nonzero adjustments.</p>
<h2>Own probabilities and secondary hybrid scores</h2>{table(secondary)}
<p>These secondary views preserve all saved probabilities without optimizing thresholds or choosing a winning mixture after seeing labels. A model's own probability, its primary audited RF score, and its work-order recommendation can disagree; they answer different reporting questions.</p>
<h2>Triage recommendations against observed labels</h2>{table(triage)}
<p>ChatGPT recommended dispatch at nine deployments (eight labelled leaks and one sound site), Claude at ten (eight leaks and two sound sites), and Gemini at eleven (eight leaks and three sound sites). These dispatch subsets have observed precisions of 88.9%, 80.0% and 72.7%, respectively, but each contains only eight of the 20 leaks. ChatGPT and Claude assigned two real leaks to no excavation; Gemini assigned four. These are small, post hoc descriptive differences in triage, not evidence of a reliable operational advantage.</p>
<p>These are saved recommendations only. “EXTEND MONITORING” is a deferral, not a correct binary prediction. A “NO EXCAVATION” recommendation at a labelled leak is reported explicitly; no abstention is silently counted as correct. No work order was executed.</p>
<h2>Dongguan and Sheffield: what was actually tested</h2>{table(pd.DataFrame(external_rows))}
<p>The earlier analysis used a separate, amplitude-normalized 13-feature RF with 300 trees and minimum leaf size 3. It is different from Paper 6's 100-tree RF using nine deployment features. Dongguan performance came from grouped five-fold out-of-fold predictions on 964 unique one-second signals. Sheffield was a fixed transfer test of that Dongguan-trained acoustic baseline on 230 unique channel-records, with no Sheffield fitting or threshold tuning. Sheffield raw units, duration and sensors differ; 202 positive channel-records and only 28 negative records are not 230 independent physical deployments.</p>
<p>Dongguan sensitivity was {100*e1['sensitivity']:.1f}% and specificity {100*e1['specificity']:.1f}%. On Sheffield these became {100*e2['sensitivity']:.1f}% and {100*e2['specificity']:.1f}%, with 16 false alarms among 28 negative channel-records and 37 misses among 202 positive channel-records. This demonstrates transfer limitations of the acoustic baseline. It provides no evidence that the three LLM hybrids behave similarly on those datasets.</p>
<p>The two public collections do not supply equivalent multi-night histories, RF reliability tables, matched training cases and operational deployment outcomes for the frozen H2 task. A fair external LLM comparison would require a separately frozen, explicitly reduced-evidence protocol and new blind model runs. We did not invent these missing inputs, carry WSD scores over to external signals, or label the RF result as ChatGPT hybrid validation. See the <a href="../Paper6_ChatGPT_blind_run/ANALYSIS_REPORT.html">external analysis report</a> for source attribution, grouping, confidence intervals and coherence results; the <a href="https://zenodo.org/records/18631450">Dongguan source</a> and <a href="https://doi.org/10.15131/shef.data.32229270.v2">Sheffield source</a> identify the datasets.</p>
<h2>Model identity and limits</h2><p>Claude Sonnet 5.5 and Gemini 4 Hydrogen are the identifiers recorded by the original project. This scoring step did not independently verify provider model revisions. The run labeled ChatGPT was carried out through fresh Codex agent sessions identified as GPT-6-based; exact backend revision and decoding settings were unavailable. For a manuscript, cite it as “GPT-6-based Codex (revision unavailable), H2 audit” until the model identity is documented. Do not identify this as a verified consumer ChatGPT model.</p>
<h2>Reproducibility and preservation</h2><p>All 15 original ChatGPT output hashes matched their pre-unsealing fingerprints before and after scoring. All 30 comparison-model source outputs, labels, RF scores, cached predictions, protocol and input packets were also fingerprinted and unchanged. No blind output was copied, edited, regenerated or replaced. Old v5 results remain intact.</p>
<p>The reproducible scoring script is <code>code/v6_score_blind.py</code>. Results are in <code>derived/v6_blind_comparison/</code>: <a href="derived/v6_blind_comparison/model_metrics.csv">full metrics</a>, <a href="derived/v6_blind_comparison/paired_comparisons.csv">paired comparisons</a>, <a href="derived/v6_blind_comparison/site_scores.csv">site scores</a>, <a href="derived/v6_blind_comparison/error_cases.csv">error cases</a>, <a href="derived/v6_blind_comparison/audit_behavior.csv">audit behavior</a>, <a href="derived/v6_blind_comparison/triage_outcomes.csv">triage outcomes</a>, <a href="derived/v6_blind_comparison/scoring_provenance.json">source fingerprints</a> and <a href="derived/v6_blind_comparison/verification.json">verification</a>. This HTML report and its figure are the review artifacts. Temporary output is kept in memory; the cleanup record lists any generated caches removed at the end. Prior source data and blind predictions are retained.</p></html>'''
    (ROOT/'Paper6_Model_Comparison.html').write_text(body,encoding='utf-8')

if __name__=='__main__': main()
