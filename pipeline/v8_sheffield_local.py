"""Sheffield-only grouped evaluation and frozen-transfer failure analysis.

All original v5/v7 artifacts are read-only inputs. New inference uses the
unchanged acoustic-only v7 prompt, schema, RF settings and audit formula.
"""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, hashlib, html, json, re, threading
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, binomtest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import v7_external_hybrids as base

ROOT=base.ROOT; EXT=base.EXT; OLD=ROOT/'derived/v7_external_hybrids'
OUT=ROOT/'derived/v8_sheffield_local'; base.OUT=OUT
FEATURES=base.FEATURES; MODELS=base.MODELS
load=base.load; save=base.save; sha=base.sha; say=base.say

def verify():
    frozen=load(OUT/'frozen_manifest.json')
    assert all(sha(OUT/f)==h for f,h in frozen['sha256'].items())
    assert sha(Path(__file__))==frozen['code_sha256']
    assert sha(ROOT/'code/v7_external_hybrids.py')==load(OLD/'frozen_manifest.json')['scoring_script_sha256']
    assert all(sha(Path(f))==h for f,h in load(OUT/'protected_sources.json').items())

def dependency_groups():
    manifest=load(EXT/'sheffield_record_manifest.json')
    paths=list((EXT/'data/sheffield').rglob('*.csv'))
    byhash={sha(p):p for p in paths}
    conditions=sorted({Path(r['file']).parent.name for r in manifest})
    parents={c:c for c in conditions}; signals=defaultdict(list)
    def find(c):
        while parents[c]!=c: parents[c]=parents[parents[c]]; c=parents[c]
        return c
    for rec in manifest:
        p=EXT/'data'/rec['file']; p=p if p.exists() else byhash[rec['sha256']]
        d=pd.read_csv(p)
        for col in d.columns[1:]:
            h=hashlib.sha256(d[col].to_numpy().tobytes()).hexdigest()
            signals[h].append(dict(condition=Path(rec['file']).parent.name,channel=col,file=rec['file']))
    links=[]
    for h,rows in signals.items():
        names=sorted({r['condition'] for r in rows})
        if len(names)>1:
            for c in names[1:]: parents[find(c)]=find(names[0])
            links.append(dict(waveform_sha256=h,sources=rows))
    roots=sorted({find(c) for c in conditions})
    mapping={c:f'C{roots.index(find(c))+1:02d}' for c in conditions}
    assert len(links)==15 and len(set(mapping.values()))==6
    save(OUT/'dependency_audit.json',dict(conditions=8,dependency_groups=6,shared_signal_families=15,condition_to_component=mapping,duplicate_links=links))
    return mapping

def split_groups(df,max_folds=3):
    """Stratify whole groups; no score/performance enters fold construction."""
    negative=sorted(g for g,d in df.groupby('group') if (d.label==0).any())
    positive=sorted(set(df.group)-set(negative))
    k=min(max_folds,len(negative)); assert k>=2
    mapping={g:i%k for i,g in enumerate(negative)}
    mapping.update({g:i%k for i,g in enumerate(positive)})
    assert all(df[df.group.map(mapping)==i].label.nunique()==2 and df[df.group.map(mapping)!=i].label.nunique()==2 for i in range(k))
    return mapping,k

def reliability(train):
    mapping,k=split_groups(train)
    folds=train.group.map(mapping).to_numpy(); p=np.full(len(train),np.nan)
    for f in range(k):
        a=folds!=f; b=~a
        p[b]=base.rf().fit(train.loc[a,FEATURES],train.loc[a,'label']).predict_proba(train.loc[b,FEATURES])[:,1]
    rows=[]
    for a,b in zip([0,.2,.4,.6,.8],[.2,.4,.6,.8,1.000001]):
        mask=(p>=a)&(p<b); n=int(mask.sum())
        rows.append(dict(probability_bin=[a,min(b,1)],n=n,observed_leak_fraction=round(float(train.label.to_numpy()[mask].mean()),4) if n else None))
    return rows

def prepare():
    if (OUT/'frozen_manifest.json').exists(): verify(); say('Frozen local protocol already verified.'); return
    OUT.mkdir(parents=True,exist_ok=True)
    assert not list((OUT/'responses').glob('**/*.json'))
    protected=base.original_files()+[p for p in OLD.rglob('*') if p.is_file()]
    save(OUT/'protected_sources.json',{str(p):sha(p) for p in sorted(set(protected))})
    groups=dependency_groups()
    d=pd.read_csv(EXT/'sheffield_transfer_predictions_features.csv')
    assert len(d)==230 and d.sha256.is_unique
    d['group']=d.condition.map(groups); mapping,k=split_groups(d)
    d['fold']=d.group.map(mapping); d['site']=[f'Q{i:04d}' for i in np.random.default_rng(8201).permutation(np.arange(1,len(d)+1))]
    d['local_rf']=np.nan; index=[]; packet_number=0
    for fold in range(k):
        tr=d[d.fold!=fold].reset_index(drop=True); te=d[d.fold==fold].copy()
        assert not set(tr.group)&set(te.group) and not set(tr.sha256)&set(te.sha256)
        model=base.rf().fit(tr[FEATURES],tr.label); p=model.predict_proba(te[FEATURES])[:,1]
        d.loc[te.index,'local_rf']=p
        med=tr[FEATURES].median().to_numpy(); effects=np.empty((len(te),len(FEATURES)))
        for j,f in enumerate(FEATURES):
            temp=te[FEATURES].copy(); temp[f]=med[j]
            effects[:,j]=p-model.predict_proba(temp)[:,1]
        scaler=StandardScaler().fit(tr[FEATURES]); a=scaler.transform(tr[FEATURES]); b=scaler.transform(te[FEATURES])
        shared=dict(feature_order=FEATURES,feature_units='centroid and bandwidth divided by 4000 Hz; other spectral descriptors dimensionless; kurtosis is standardized fourth moment',training_n=len(tr),training_groups=int(tr.group.nunique()),training_only_class_medians={str(y):np.round(tr.loc[tr.label==y,FEATURES].median().to_numpy(),5).tolist() for y in [0,1]},training_only_inner_grouped_cv_reliability=reliability(tr))
        cases=[]
        for i,(_,r) in enumerate(te.iterrows()):
            dist=np.sqrt(((a-b[i])**2).sum(axis=1)); near=np.argsort(dist,kind='stable')[:5]; top=np.argsort(-np.abs(effects[i]),kind='stable')[:4]
            cases.append(dict(site=r.site,rf_probability=round(float(p[i]),6),features=np.round(r[FEATURES].to_numpy(dtype=float),5).tolist(),rf_median_replacement_effects=[dict(feature=FEATURES[j],training_median=round(float(med[j]),5),effect=round(float(effects[i,j]),5)) for j in top],nearest_training_references=[dict(distance=round(float(dist[t]),4),known_leak_condition=int(tr.iloc[t].label),features=np.round(tr.iloc[t][FEATURES].to_numpy(dtype=float),5).tolist()) for t in near]))
        order=np.random.default_rng(8202+fold).permutation(len(cases))
        for start in range(0,len(order),24):
            packet_number+=1; name=f'P{packet_number:03d}'; targets=[cases[i] for i in order[start:start+24]]
            save(OUT/'packets'/f'{name}.json',dict(instructions=base.INSTRUCTIONS,shared_training_evidence=shared,targets=targets))
            index.append(dict(packet=name,fold=fold,sites=[r['site'] for r in targets]))
        say('Prepared local fold',fold,'held-out groups',te.group.nunique(),'cases',len(te),'training groups',tr.group.nunique())
    assert np.isfinite(d.local_rf).all()
    d.to_csv(OUT/'private_local_index.csv',index=False)
    save(OUT/'packet_index.json',index)
    protocol=dict(frozen_utc=base.now(),version='Sheffield-only grouped acoustic H2 v1',models=MODELS,model_choice='User requested Opus 5.5, API returned 404; user explicitly selected keep Sonnet 5.5. No Opus model substituted or relabelled.',task=base.INSTRUCTIONS,unit='230 unique source channel-records; not 230 independent acquisitions',training='Sheffield only, never Dongguan; three outer grouped folds, every tool/ref/reliability estimate excludes held-out dependency groups',groups='Union experimental conditions linked by any duplicated raw channel before deduplication; 8 conditions become 6 connected components',outer_split='Three dependency-group folds, one of three background-bearing components in each fold; other components distributed round-robin deterministically. Uses labels only for stratification, not performance.',outer_fold_by_group=mapping,inner_reliability='Two group folds in each outer-training set, preserving a background-bearing group on either side',rf='Same 13 normalized descriptors, 300 trees, min_samples_leaf=3, balanced class weights, seed 42 as frozen v7',primary_score='sigmoid(logit(clip(p_RF,0.02,0.98))+0.7*a)',threshold=.5,references=5,inference='Same prompt/schema/provider settings and fresh-session calls as v7. At most two requests per model concurrently; no tool calls.',retries=load(OLD/'PROTOCOL.json')['retry'],analysis='Score only after all three models complete and responses are locked. Cluster bootstrap 2000 draws by six dependency groups, seed 7103; one-class draws excluded. Report pooled out-of-fold AUC and threshold metrics; small-fold/class estimates are descriptive.',limitations=['Exploratory post hoc variant requested after transfer outcomes were observed; predictor target labels remain blinded.','Only three background-bearing acquisition components; most held-out folds have very few independent negative acquisitions.','This is within-laboratory discrimination, not independent field validation or improved cross-project transfer.','Record condition label is not independently verified leak audibility at every sensor.','No test-fold-based threshold, feature, prompt, model or hyperparameter selection.'],frozen_prior_versions='All original v5/v7 predictions and other v7 artifacts fingerprinted and retained unchanged',temperature='0 requested for Poe; Codex temperature not exposed',reasoning='low requested; wrappers are not identical')
    save(OUT/'PROTOCOL.json',protocol)
    paths=[OUT/'PROTOCOL.json',OUT/'private_local_index.csv',OUT/'packet_index.json',OUT/'dependency_audit.json',OUT/'protected_sources.json']+list((OUT/'packets').glob('*.json'))
    save(OUT/'frozen_manifest.json',dict(frozen_utc=base.now(),sha256={str(p.relative_to(OUT)):sha(p) for p in paths},code_sha256=sha(Path(__file__))))
    verify(); say('FROZEN',len(index),'packets; 230 held-out Sheffield cases per model.')

def run():
    verify(); index=load(OUT/'packet_index.json')
    jobs=[(m,p['packet']) for p in index for m in MODELS if not (OUT/'responses'/m/(p['packet']+'.json')).exists()]
    limits={m:threading.Semaphore(2) for m in MODELS}; stop=threading.Event()
    start=base.balance()
    def task(m,p):
        with limits[m]:
            if stop.is_set(): return m,p,'not_submitted_after_failure'
            if m!='codex-gpt-6.1-sol' and base.balance()<15000: stop.set(); return m,p,'balance_reserve'
            r=base.job(m,p)
            if r[2] not in ['ok','cached']: stop.set()
            return r
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(task,*j) for j in jobs]
        for n,f in enumerate(as_completed(futures),1): say('LOCAL',n,'/',len(jobs),*f.result())
    save(OUT/'run_ledger.json',dict(finished_utc=base.now(),poe_start_balance=start,poe_end_balance=base.balance(),response_counts={m:len(list((OUT/'responses'/m).glob('*.json'))) for m in MODELS}))
    verify()

def diagnose():
    """Descriptive forensics on frozen v7; no model or threshold tuned here."""
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(OLD/'scored_predictions.csv')
    dg=pd.read_csv(EXT/'dongguan_predictions_features.csv'); sh=pd.read_csv(EXT/'sheffield_transfer_predictions_features.csv')
    evidence={t['site']:t for p in (OLD/'packets').glob('*.json') for t in load(p)['targets']}
    d['nearest_distance']=d.site.map(lambda s:min(r['distance'] for r in evidence[s]['nearest_training_references']))
    d['neighbor_leak_fraction']=d.site.map(lambda s:np.mean([r['known_leak_condition'] for r in evidence[s]['nearest_training_references']]))
    threshold=float(d.loc[d.dataset=='dongguan','nearest_distance'].quantile(.95))
    summary=dict(source_oof_nn_distance_95th_percentile=threshold,source_median_nn_distance=float(d.loc[d.dataset=='dongguan','nearest_distance'].median()),target_median_nn_distance=float(d.loc[d.dataset=='sheffield','nearest_distance'].median()),target_fraction_beyond_source_95th=float((d.loc[d.dataset=='sheffield','nearest_distance']>threshold).mean()),audit_minus_one_positive_flip_boundary=float(1/(1+np.exp(-.7))),caveat='Support threshold is a descriptive source-only diagnostic, not a tuned target policy or newly validated transfer improvement.')
    stats=[]
    for feature in FEATURES:
        source=dg[feature].to_numpy(); target=sh[feature].to_numpy(); q1,q99=np.quantile(source,[.01,.99]); iqr=np.subtract(*np.percentile(source,[75,25]))
        stats.append(dict(feature=feature,source_median=float(np.median(source)),target_median=float(np.median(target)),target_minus_source_median_in_source_IQR=float((np.median(target)-np.median(source))/max(iqr,1e-12)),KS_statistic=float(ks_2samp(source,target).statistic),target_outside_source_1_99_percentile=float(((target<q1)|(target>q99)).mean())))
    pd.DataFrame(stats).sort_values('KS_statistic',ascending=False).to_csv(OUT/'transfer_feature_shift.csv',index=False)
    transfer=d[d.dataset=='sheffield'].merge(sh[['sha256']+FEATURES],on='sha256',validate='one_to_one')
    rows=[]; reason_rows=[]; examples=[]
    for model in MODELS:
        a=transfer['audit_'+model]; p=transfer['primary_'+model]; correct_rf=(transfer.rf_probability>=.5)==transfer.label; correct_h=(p>=.5)==transfer.label
        for (condition,y),g in transfer.assign(adjusted=a!=0,broken=correct_rf&~correct_h,fixed=~correct_rf&correct_h).groupby(['group','label']):
            rows.append(dict(model=model,condition=condition,label=y,n=len(g),adjusted=int(g.adjusted.sum()),classification_broken=int(g.broken.sum()),classification_fixed=int(g.fixed.sum())))
        for i,r in transfer[a!=0].iterrows():
            reason=r['reason_'+model]
            reason_rows.append(dict(model=model,site=r.site,condition=r.group,channel=r.channel,label=int(r.label),rf_probability=r.rf_probability,primary_probability=r['primary_'+model],adjustment=int(r['audit_'+model]),nearest_distance=r.nearest_distance,source_nn_95_exceeded=bool(r.nearest_distance>threshold),neighbor_leak_fraction=r.neighbor_leak_fraction,reason=reason,tonal_or_narrow=bool(re.search(r'tonal|narrow|peak|concentrat',reason,re.I)),reference_conflict=bool(re.search(r'refer|neighbo|analogu|training|case',reason,re.I))))
        broken=transfer[correct_rf&~correct_h]
        summary[model]=dict(adjusted=int((a!=0).sum()),negative_adjustments=int((a<0).sum()),true_leaks_downgraded=int(((a<0)&(transfer.label==1)).sum()),broken=int((correct_rf&~correct_h).sum()),fixed=int((~correct_rf&correct_h).sum()),downgraded_leak_outside_source_nn95=int(((a<0)&(transfer.label==1)&(transfer.nearest_distance>threshold)).sum()),broken_rf_probability_min=float(broken.rf_probability.min()) if len(broken) else None,broken_rf_probability_max=float(broken.rf_probability.max()) if len(broken) else None)
        if len(broken):
            r=broken.iloc[0]; examples.append(dict(model=model,site=r.site,condition=r.group,channel=r.channel,rf_probability=r.rf_probability,primary_probability=r['primary_'+model],reason=r['reason_'+model],source_label=int(r.label)))
    pd.DataFrame(rows).to_csv(OUT/'transfer_condition_errors.csv',index=False)
    pd.DataFrame(reason_rows).to_csv(OUT/'transfer_adjusted_case_review.csv',index=False)
    save(OUT/'transfer_diagnosis.json',summary); save(OUT/'transfer_error_examples.json',examples)
    say(json.dumps(summary,indent=2))

def score():
    verify(); packet_index=load(OUT/'packet_index.json'); expected=len(packet_index)
    paths=list((OUT/'responses').glob('*/*.json'))
    assert all(len(list((OUT/'responses'/m).glob('*.json')))==expected for m in MODELS)
    lock=OUT/'locked_responses.json'
    if lock.exists(): assert all(sha(OUT/f)==h for f,h in load(lock)['sha256'].items())
    else: save(lock,dict(locked_utc=base.now(),sha256={str(p.relative_to(OUT)):sha(p) for p in paths}))
    d=pd.read_csv(OUT/'private_local_index.csv'); y=d.label.to_numpy(); raw=d.local_rf.to_numpy()
    probs={'RF tool':raw,'RF zero-audit clipped control':np.clip(raw,.02,.98)}
    behavior=[]
    for m in MODELS:
        docs=[load(f) for f in (OUT/'responses'/m).glob('*.json')]
        assert sum(len(x['results']) for x in docs)==230
        records={r['site']:r for x in docs for r in x['results']}; assert set(records)==set(d.site)
        assert all(x['metadata'].get('tool_event_count',x['metadata'].get('tool_call_count',0))==0 for x in docs)
        if m!='codex-gpt-6.1-sol': assert all(x['metadata']['returned_model']==m for x in docs)
        a=d.site.map(lambda s:records[s]['audit_adjustment']).to_numpy(); own=d.site.map(lambda s:records[s]['leak_probability']).to_numpy()
        q=np.clip(raw,.02,.98); primary=1/(1+np.exp(-(np.log(q/(1-q))+.7*a)))
        d['audit_'+m]=a; d['own_'+m]=own; d['primary_'+m]=primary; d['reason_'+m]=d.site.map(lambda s:records[s]['audit_reason'])
        probs[m+' | primary']=primary; probs[m+' | own']=own; probs[m+' | mean']=(raw+own)/2
        old_correct=(raw>=.5)==y; new_correct=(primary>=.5)==y
        behavior.append(dict(model=m,adjusted=int((a!=0).sum()),negative=int((a<0).sum()),downgraded_true_leaks=int(((a<0)&(y==1)).sum()),classification_fixed=int((~old_correct&new_correct).sum()),classification_broken=int((old_correct&~new_correct).sum())))
    d.to_csv(OUT/'local_scored_predictions.csv',index=False)
    boot=base.bootstrap(y,probs,d.group.to_numpy()); results=[]; pairs=[]
    for name,p in probs.items():
        lo,hi=np.percentile(boot[name],[2.5,97.5]); results.append(dict(model=name,n=len(d),AUC_cluster_lo=lo,AUC_cluster_hi=hi,**base.metrics(y,p)))
    for m in MODELS:
        name=m+' | primary'; diff=boot[name]-boot['RF tool']; lo,hi=np.percentile(diff,[2.5,97.5]); a=(probs[name]>=.5)==y; b=(raw>=.5)==y; win=int((a&~b).sum()); lose=int((~a&b).sum())
        pairs.append(dict(model=name,delta_AUC=float(roc_auc_score(y,probs[name])-roc_auc_score(y,raw)),delta_AUC_cluster_lo=lo,delta_AUC_cluster_hi=hi,classification_fixed=win,classification_broken=lose,McNemar_two_sided=float(binomtest(win,win+lose,.5).pvalue) if win+lose else 1.,valid_bootstrap=len(diff)))
    m=pd.DataFrame(results); p=pd.DataFrame(pairs); b=pd.DataFrame(behavior)
    m.to_csv(OUT/'local_metrics.csv',index=False); p.to_csv(OUT/'local_paired_comparisons.csv',index=False); b.to_csv(OUT/'local_audit_behavior.csv',index=False)
    report(m,p,b); verify()
    save(OUT/'verification.json',dict(original_sources_unchanged=True,frozen_protocol_and_code_unchanged=True,cases_per_model=230,response_files=len(paths),target_labels_withheld=True,training_labels_supplied_only_for_training_references=True,zero_recorded_tool_calls=True,groups_disjoint_in_all_tools=True))
    say(m[(m.model=='RF tool')|m.model.str.endswith(' | primary')].to_string(index=False))

def report(m,p,b):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    diag=load(OUT/'transfer_diagnosis.json'); shifted=pd.read_csv(OUT/'transfer_feature_shift.csv'); review=pd.read_csv(OUT/'transfer_adjusted_case_review.csv')
    old=pd.read_csv(OLD/'metrics.csv'); transfer=old[(old.dataset=='sheffield')&((old.model=='RF tool')|old.model.str.endswith(' | primary'))].copy(); transfer['evaluation']='Dongguan → Sheffield transfer'
    local=m[(m.model=='RF tool')|m.model.str.endswith(' | primary')].copy(); local['evaluation']='Sheffield-only grouped holdout'
    comparison=pd.concat([transfer,local],ignore_index=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,(name,rows) in zip(axes,comparison.groupby('evaluation')):
        ax.bar(np.arange(len(rows)),rows.AUC,color=['#347d9c','#3b9c77','#bd7c35','#8c6eaa']); ax.set(xticks=np.arange(len(rows)),xticklabels=rows.model.str.replace(' | primary','',regex=False),ylim=(0,1),ylabel='AUC',title=name); ax.tick_params(axis='x',rotation=30)
        for i,v in enumerate(rows.AUC): ax.text(i,v+.015,f'{v:.3f}',ha='center')
    fig.savefig(OUT/'transfer_vs_local.png',dpi=180); plt.close(fig)
    def table(df): return df.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}')
    examples=load(OUT/'transfer_error_examples.json')
    primary_columns=['evaluation','model','AUC','AUC_cluster_lo','AUC_cluster_hi','balanced_accuracy','sensitivity','specificity','FP','FN','Brier']
    mechanism=[]
    for model in MODELS:
        s=diag[model]; r=review[review.model==model]
        mechanism.append(dict(model=model,negative_audits=s['negative_adjustments'],true_leaks_downgraded=s['true_leaks_downgraded'],RF_correct_changed_to_wrong=s['broken'],RF_wrong_fixed=s['fixed'],downgraded_leaks_outside_source_support=s['downgraded_leak_outside_source_nn95'],tonal_narrow_keywords=int(r.tonal_or_narrow.sum()),reference_keywords=int(r.reference_conflict.sum())))
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Sheffield transfer diagnosis and local evaluation</title><style>body{{font:16px/1.55 system-ui;max-width:1450px;margin:36px auto;padding:0 24px;color:#20313c}}h1,h2{{color:#175873}}table{{border-collapse:collapse;display:block;overflow:auto;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #d0dbe1;text-align:left}}th{{background:#edf4f7}}img{{max-width:100%}}.lead{{background:#edf5fa;padding:18px;border-left:5px solid #286c91}}</style>
<h1>Why the Sheffield hybrids failed in transfer, and a Sheffield-only version</h1><p>24 November 2026. Original Paper 6 and v7 predictions are unchanged. Opus 5.5 returned API 404; the user selected continued use of Sonnet 5.5. No other Opus version was substituted.</p>
<p class="lead">The transfer hybrids mainly suppressed true leak-condition recordings. The new version trains and tests within Sheffield using held-out dependency groups. It answers a different question—within-laboratory generalization—and cannot by itself establish improved cross-project transfer.</p>
<h2>Frozen transfer failure mechanism</h2>{table(pd.DataFrame(mechanism))}
<p>A negative audit step subtracts 0.7 from the RF log-odds. Any initially positive RF probability between 0.5 and {diag['audit_minus_one_positive_flip_boundary']:.4f} therefore becomes negative after a −1 audit. This fixed numerical rule explains how modest distrust creates many extra misses; it does not mean the RF had strongly established a leak. Audit text and case-level outputs are in transfer_adjusted_case_review.csv; keyword counts above are descriptive coding, not independent causal findings.</p>
{table(pd.DataFrame(examples))}
<h2>Distribution and evidence mismatch</h2><p>The source out-of-group 95th percentile nearest-reference distance was {diag['source_oof_nn_distance_95th_percentile']:.4f}. Median nearest-reference distance was {diag['source_median_nn_distance']:.4f} for Dongguan and {diag['target_median_nn_distance']:.4f} for Sheffield; {100*diag['target_fraction_beyond_source_95th']:.1f}% of Sheffield cases exceed that source threshold. Distances are those already supplied in the frozen training-only packets. This is a source-only support diagnostic, not an optimized target threshold or proof of reliable shift detection.</p>
{table(shifted)}
<p>Dongguan mixes source materials/devices and one-second clips, whereas Sheffield uses calibrated MDPE laboratory recordings and descriptors averaged across one-second windows of longer channel-records. RF predictions on mean descriptors need not equal mean predictions on windows. Different materials, sensors, experimental conditions, aggregation and class composition plausibly change feature-label relationships. The observed marginal shifts and rationale text support these explanations but do not isolate their causal contributions. An acquisition's leak-condition label also does not guarantee locally audible leakage at every distant sensor.</p>
<p>The reliability tables and nearest labeled cases were valid for Dongguan training folds; they were not validated Sheffield calibration. Under shift, feature similarity does not guarantee the same outcome relationship. The LLMs sometimes treated source-reference conflict or spectral concentration as evidence against the source leak condition despite the prompt's cautions. This is consistent with the general risks studied in <a href="https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html">Ovadia et al. (2019)</a> and <a href="https://proceedings.mlr.press/v139/koh21a.html">WILDS (Koh et al., 2021)</a>; those studies do not establish the particular cause in these acoustic data.</p>
<h2>How to improve transfer without tuning Sheffield test labels</h2><ol>
<li><strong>Preserve RF under unsupported auditing.</strong> Before touching a target test set, validate override rules on source-only, held-out sensor/material/recording groups. Permit automatic adjustments only if an agreed source validation criterion is met; otherwise retain the RF score and show the LLM rationale as an advisory note. This prevents incremental audit damage but cannot repair an already shifted RF. A source-trained support gate can flag unsupported inputs; it must not be assumed to detect conditional shift.</li>
<li><strong>Match the signal-processing unit.</strong> Predefine common resampling, frequency bands and one-second extraction for both collections. Compare window-probability aggregation versus feature averaging using source training groups only, freeze the choice, then apply it once to target recordings. Keep all windows/channels from a source acquisition together in validation.</li>
<li><strong>Use source-domain stress tests.</strong> Leave source sensor types and pipe materials out during development, then choose robust descriptors and reference/reliability scope from those source folds. Add genuinely independent training data covering target-like sensors/materials rather than treating nearest numerical neighbors as validated target cases.</li>
<li><strong>Separate calibration from discrimination.</strong> Fit any sigmoid or isotonic calibration and decision thresholds on source training/validation groups inside the outer split. Source calibration is not guaranteed to survive shift; do not recalibrate on the current Sheffield test labels. <a href="https://proceedings.mlr.press/v70/guo17a.html">Guo et al. (2017)</a> explains probability calibration, primarily for neural networks; no guarantee for this RF or target setting is implied.</li>
<li><strong>Use independent target development data if local adaptation is needed.</strong> Collect new, separate Sheffield-like leak and background acquisitions, develop on them, and reserve new entire acquisitions as the final holdout. The current within-Sheffield experiment is a separately disclosed local benchmark, not a test-label-tuned improvement of the frozen transfer run.</li></ol>
<p>These proposals are not selected by maximizing Sheffield test metrics. No old probability, audit, threshold or score has been rewritten. A future gated/calibrated transfer version should be frozen on source validation or new development data before its final test.</p>
<h2>Sheffield-only protocol and results</h2><p>Raw channel hashes show 15 duplicated signal families linking condition 2 with 5 and condition 3 with 4. Merging linked conditions leaves six dependency components. Three contain distinct background acquisitions. Three outer group folds each hold out a background-bearing component and a leak-only component; all RF fitting, training medians, retrieval, feature explanations and two-fold inner reliability estimates exclude the held-out components. No Dongguan data enter the local model or reference tools.</p>
<p>There are only 28 negative channel-records and three independent background-bearing components. Channels within an acquisition are dependent; 230 is not the number of independent experiments. Group bootstrap intervals condition on trained models and one inference run, exclude resamples without both classes, and do not resolve the small-control limitation. Fold construction used labels for stratification only, not to select favorable scores. This local design was requested after seeing transfer outcomes, so it is exploratory.</p>
{table(comparison[primary_columns])}<img src="transfer_vs_local.png" alt="Transfer versus within-Sheffield grouped model AUCs">
<h2>Local paired differences and audit behavior</h2>{table(p)}{table(b)}
<p>Positive AUC differences are local hybrid minus its common local RF tool. Paired intervals spanning zero do not establish a gain. McNemar is a nominal, unadjusted exact two-sided test on channel-record discordances; dependence and few acquisition groups limit interpretation.</p>
<h2>All local secondary probabilities</h2>{table(m)}
<h2>Artifacts, preservation and cleanup</h2><p>Frozen local protocol: <a href="PROTOCOL.json">PROTOCOL.json</a>. Group audit: <a href="dependency_audit.json">dependency_audit.json</a>. <a href="local_metrics.csv">Local metrics</a>, <a href="local_scored_predictions.csv">per-case scores/reasons</a>, <a href="transfer_diagnosis.json">transfer diagnosis</a>, and <a href="verification.json">verification</a> are retained with immutable packets and model responses. Earlier v5/v7 predictions and supporting v7 artifacts are fingerprinted in protected_sources.json. Only generated bytecode and the empty task scratch directory are removed after analysis. Source collections remain <a href="https://zenodo.org/records/18631450">Dongguan</a> and <a href="https://doi.org/10.15131/shef.data.32229270.v2">Sheffield</a>, CC BY 4.0.</p></html>'''
    (OUT/'REPORT.html').write_text(body,encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['prepare','run','diagnose','score']); args=parser.parse_args(); globals()[args.command]()
