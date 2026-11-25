"""Frozen, reduced-evidence external H2 experiment. Never edits original runs.

Commands: prepare, run, status, score. Credentials are read in memory only.
"""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import argparse, hashlib, html, json, os, re, subprocess, sys, tempfile, threading, time
import numpy as np
import pandas as pd
import requests
from scipy.stats import rankdata, binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss

ROOT=Path(__file__).resolve().parents[1]
BLIND=ROOT.parent/'Paper6_ChatGPT_blind_run'
EXT=BLIND/'external_validation'
OUT=ROOT/'derived/v7_external_hybrids'
FEATURES=['centroid','bandwidth','flatness','entropy','peak_fraction','zcr','kurtosis','band_20_100','band_100_250','band_250_500','band_500_1000','band_1000_2000','band_2000_3900']
MODELS={'codex-gpt-6.1-sol':'gpt-6.1-sol','claude-sonnet-5.5':'claude-sonnet-5.5','gemini-4-hydrogen':'gemini-4-hydrogen'}
MAX_CASES=24
INSTRUCTIONS='''You audit a random-forest (RF) leak-condition probability using acoustic evidence. This is a blind acoustic-only experiment. Judge each target independently. Use only the provided packet; do not use tools, web, files, memory or outside datasets. Target filenames, collection identity and labels are deliberately hidden. Training references may have known outcomes but no target has a supplied outcome.
The target is whether the source recording was made under a leak-generating condition versus a no-leak/background condition. An apparently quiet or tonal channel can still have been recorded during a leak. Do not reinterpret this as verified leak audibility at a field logger. Features are amplitude-normalized spectral descriptors; absolute amplitude, sensor coupling, multi-night persistence, temporal/transient summaries, spatial distances and operational outcomes are unavailable. Do not invent them or infer persistence from spectral shape. A source channel may summarize multiple one-second windows; that does not establish multi-night persistence.
Feature order and units are supplied. For each target, RF probability comes from training data excluding its held-out group/fold, or from a separate source collection in transfer testing. Training-only class medians, inner grouped-CV reliability bins, five nearest training references, and the four largest median-replacement RF probability effects are supplied. Positive effect means the actual feature raised the score relative to replacing it with its training median. Explanations are model sensitivities, not causal effects; features can be correlated. Reference similarity and reliability may fail under domain shift. No single frequency band, low flatness or spectral tonality establishes or rules out a leak.
Audit the RF; do not replace it. Set audit_adjustment to 0 when evidence agrees, is mixed, or lacks needed context. Use -1/+1 only for a clear specific contradiction and -2/+2 only for several independent contradictions. Most targets should receive 0. Supply your own leak-condition probability as a secondary output. Primary score will be sigmoid(logit(clip(p_RF,0.02,0.98))+0.7*a); the threshold 0.5 is fixed. Do not optimize thresholds or try to match a class balance.
Return ONLY one JSON object with key "results", containing exactly one object per target and exactly these keys: "site" (supplied ID), "audit_adjustment" (integer -2,-1,0,1,2), "leak_probability" (number 0 to 1), "verdict" ("LEAK", "NO LEAK" or "AMBIGUOUS"), and "audit_reason" (one concise sentence, at most 25 whitespace-separated words). No markdown, commentary, additional keys or tool calls.'''
OPTIONS={'claude-sonnet-5.5':dict(web_search=False,effort='low',thinking_display=False),'gemini-4-hydrogen':dict(web_search=False,thinking_level='low')}
PRINT_LOCK=threading.Lock()

def say(*x):
    with PRINT_LOCK: print(*x,flush=True)
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def textsha(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def rf(): return RandomForestClassifier(n_estimators=300,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=-1)
def auth():
    import os
    return os.environ["POE_API_KEY"]  # set in the shell; never stored in the repository
def balance():
    r=requests.get('https://api.poe.com/usage/current_balance',headers={'Authorization':'Bearer '+auth()},timeout=45)
    r.raise_for_status(); return r.json()['current_point_balance']
def original_files():
    files=list((BLIND/'outputs').glob('chatgpt_H2_*.json'))
    files+=list((ROOT/'derived/v5/llm_raw').glob('*_H2_*.json'))
    files += [ROOT/'derived/v5/predictions_v5.csv',ROOT/'derived/v5/benchmark_v5.csv',BLIND/'validation_checks.json',EXT/'dongguan_predictions_features.csv',EXT/'sheffield_transfer_predictions_features.csv',EXT/'results.json']
    return sorted(set(files))
def verify_originals():
    snap=load(OUT/'protected_sources.json')
    assert all(sha(Path(p))==h for p,h in snap.items()),'Protected source changed'
    old=load(BLIND/'validation_checks.json')
    assert all(sha(BLIND/'outputs'/f"chatgpt_H2_{c['zone']}.json")==c['sha256'] for c in old)

def reliability(train):
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=7102)
    p=np.full(len(train),np.nan)
    for a,b in cv.split(train[FEATURES],train.label,train.group):
        assert not set(train.iloc[a].group)&set(train.iloc[b].group)
        p[b]=rf().fit(train.iloc[a][FEATURES],train.iloc[a].label).predict_proba(train.iloc[b][FEATURES])[:,1]
    rows=[]
    for a,b in zip([0,.2,.4,.6,.8],[.2,.4,.6,.8,1.000001]):
        mask=(p>=a)&(p<b); n=int(mask.sum())
        rows.append(dict(probability_bin=[a,min(b,1)],n=n,observed_leak_fraction=round(float(train.label.to_numpy()[mask].mean()),4) if n else None))
    return rows

def prepare():
    if (OUT/'frozen_manifest.json').exists():
        verify_frozen(); say('Existing frozen protocol and packets verified; preparation not repeated.'); return
    OUT.mkdir(parents=True,exist_ok=True)
    assert not list((OUT/'responses').glob('**/*.json')),'Cannot rebuild after inference'
    dg=pd.read_csv(EXT/'dongguan_predictions_features.csv'); sh=pd.read_csv(EXT/'sheffield_transfer_predictions_features.csv')
    assert len(dg)==964 and len(sh)==230
    assert not dg.sha256.duplicated().any() and not sh.sha256.duplicated().any()
    assert (dg.groupby('group').fold.nunique()==1).all()
    assert not set(dg.sha256)&set(sh.sha256)
    save(OUT/'protected_sources.json',{str(p):sha(p) for p in original_files()})
    # Random IDs are allocated to all cases together, independently of outcome.
    rng=np.random.default_rng(7101); ids=[f'X{x:04d}' for x in rng.permutation(np.arange(1,len(dg)+len(sh)+1))]
    dg['site']=ids[:len(dg)]; sh['site']=ids[len(dg):]
    targets=[]; packets=[]; manifest=[]; packet_counter=0
    for fold in list(range(5))+['transfer']:
        train=dg.copy() if fold=='transfer' else dg[dg.fold!=fold].copy()
        test=sh.copy() if fold=='transfer' else dg[dg.fold==fold].copy()
        train=train.reset_index(drop=True); test=test.reset_index(drop=True)
        if fold!='transfer': assert not set(train.group)&set(test.group)
        assert not set(train.sha256)&set(test.sha256)
        model=rf().fit(train[FEATURES],train.label)
        p=model.predict_proba(test[FEATURES])[:,1]
        assert np.allclose(p,test.probability.to_numpy(),rtol=0,atol=1e-13),f'Existing RF tool not reproduced: {fold}'
        med=train[FEATURES].median().to_numpy()
        effects=np.empty((len(test),len(FEATURES)))
        for j,f in enumerate(FEATURES):
            temp=test[FEATURES].copy(); temp[f]=med[j]
            effects[:,j]=p-model.predict_proba(temp)[:,1]
        sc=StandardScaler().fit(train[FEATURES]); source=sc.transform(train[FEATURES]); dest=sc.transform(test[FEATURES])
        shared=dict(feature_order=FEATURES,feature_units='centroid and bandwidth divided by 4000 Hz; flatness, entropy, maximum-bin fraction, zero-crossing rate and band fractions dimensionless; kurtosis is standardized fourth moment',training_n=len(train),training_groups=int(train.group.nunique()),training_only_class_medians={str(y):np.round(train.loc[train.label==y,FEATURES].median().to_numpy(),5).tolist() for y in [0,1]},training_only_inner_grouped_cv_reliability=reliability(train))
        case_rows=[]
        for i,row in test.iterrows():
            distance=np.sqrt(((source-dest[i])**2).sum(axis=1)); near=np.argsort(distance,kind='stable')[:5]
            top=np.argsort(-np.abs(effects[i]),kind='stable')[:4]
            case_rows.append(dict(site=row.site,rf_probability=round(float(p[i]),6),features=np.round(row[FEATURES].to_numpy(dtype=float),5).tolist(),rf_median_replacement_effects=[dict(feature=FEATURES[j],training_median=round(float(med[j]),5),effect=round(float(effects[i,j]),5)) for j in top],nearest_training_references=[dict(distance=round(float(distance[k]),4),known_leak_condition=int(train.iloc[k].label),features=np.round(train.iloc[k][FEATURES].to_numpy(dtype=float),5).tolist()) for k in near]))
            targets.append(dict(site=row.site,dataset='sheffield' if fold=='transfer' else 'dongguan',fold=str(fold),label=int(row.label),group=row.condition if fold=='transfer' else row.group,source_file=row.file,channel=row.get('channel',''),sha256=row.sha256,rf_probability=float(p[i])))
        order=rng.permutation(len(case_rows))
        for start in range(0,len(order),MAX_CASES):
            packet_counter+=1; packet_id=f'P{packet_counter:03d}'
            cases=[case_rows[k] for k in order[start:start+MAX_CASES]]
            packet=dict(instructions=INSTRUCTIONS,shared_training_evidence=shared,targets=cases)
            path=OUT/'packets'/f'{packet_id}.json'; save(path,packet)
            packets.append(path); manifest.append(dict(packet=packet_id,sites=[c['site'] for c in cases],fold=str(fold),dataset='sheffield' if fold=='transfer' else 'dongguan'))
        say('Prepared held-out fold',fold,'targets',len(test),'training groups',train.group.nunique())
    pd.DataFrame(targets).to_csv(OUT/'private_target_index.csv',index=False)
    save(OUT/'packet_index.json',manifest)
    metadata=requests.get('https://api.poe.com/v1/models',headers={'Authorization':'Bearer '+auth()},timeout=60); metadata.raise_for_status()
    save(OUT/'provider_model_metadata.json',[m for m in metadata.json()['data'] if m['id'] in OPTIONS])
    protocol=dict(frozen_utc=now(),version='external_H2_acoustic_only_v1',status='Frozen before external hybrid outcomes; prior acoustic RF results and source labels already known to coordinator',n_dongguan=964,n_sheffield=230,unit='Dongguan unique 1-second waveform; Sheffield unique channel-record (mean descriptors of full 1-second windows)',models=MODELS,providers={'codex-gpt-6.1-sol':'Codex CLI, fresh ephemeral sessions','claude-sonnet-5.5':'Poe stateless Chat Completions','gemini-4-hydrogen':'Poe stateless Chat Completions'},reasoning='Low effort requested for all; provider meanings are not equivalent',temperature='0 requested for Poe; Codex temperature not exposed',max_targets_per_packet=MAX_CASES,prompt=INSTRUCTIONS,original_task_difference='No multi-night persistence, absolute amplitude, spatial peer context, operational context, or work orders. Reduced-evidence external protocol, not a replication of the full WSD H2 deployment task.',primary_formula='sigmoid(logit(clip(rf_probability,0.02,0.98))+0.7*a)',secondary_formula='(rf_probability+LLM_probability)/2',threshold=.5,rf=dict(trees=300,min_samples_leaf=3,class_weight='balanced',random_state=42,features=FEATURES),dongguan='Exactly the existing five grouped folds, each target group excluded from RF training, explanations, retrieval, class profiles and inner grouped reliability',sheffield='Strict cross-project transfer: train RF/reference tools on all Dongguan only. No Sheffield labels are used in any training, retrieval, reliability estimate, model selection or threshold setting.',references=5,reliability_inner_folds=5,request_context='Fresh single request/session per packet. Target filenames, dataset names, ground-truth labels, folds, group IDs and other model outputs excluded.',retry='Up to two transport retries for transient failure with identical payload. One additional fresh request only for format failure, retaining failure record. Never retry for an unfavorable scientific outcome.',scoring='Score only after all model responses are locked; complete matched case set. Report AUC, balanced accuracy, sensitivity, specificity, precision, F1, Brier, FP/FN, cluster bootstrap paired AUC differences and exact McNemar; no hyperparameter or threshold tuning.',bootstrap='2000 paired resamples of Dongguan apparent source groups or Sheffield experimental conditions, seed 7103; no model refitting',sampling='Complete census of all 1194 deduplicated cases; no performance-based subsampling',limitations=['Coordinator has already observed original acoustic baseline results; model target labels are blinded, but study is exploratory.','Filename-derived Dongguan groups are not independently verified sites.','Public data may have been in LLM pretraining; hiding filenames and using numerical evidence reduces identity cues, not proof of pretraining exclusion.','Sheffield source leak-condition label need not imply audible leak at every sensor.','Different provider wrappers/system prompts and stochastic generation can affect comparison.','One inference per model per case; no run-to-run uncertainty estimate.'],api_options=OPTIONS,codex_arguments=['--ignore-user-config','--skip-git-repo-check','--ephemeral','--sandbox','read-only','--model','gpt-6.1-sol','--config','model_reasoning_effort=low','--json'],codex_tool_policy='Prompt forbids all tools. JSONL event stream audited; any tool event invalidates run.',poe_tool_policy='tools=[], tool_choice=none and web_search=false requested; reject any response tool calls',budget='Use existing subscription/points only, no purchase; stop Poe submissions if remaining points below 15000; preserve incomplete runs without inventing results')
    save(OUT/'PROTOCOL.json',protocol)
    locked=[OUT/'PROTOCOL.json',OUT/'private_target_index.csv',OUT/'packet_index.json',OUT/'protected_sources.json',OUT/'provider_model_metadata.json']+packets
    save(OUT/'frozen_manifest.json',dict(frozen_utc=now(),sha256={str(p.relative_to(OUT)):sha(p) for p in locked},scoring_script_sha256=sha(Path(__file__))))
    verify_frozen(); say('FROZEN',len(targets),'targets;',len(packets),'packets. Starting balance',balance())

def verify_frozen():
    frozen=load(OUT/'frozen_manifest.json')
    assert all(sha(OUT/p)==h for p,h in frozen['sha256'].items()),'Frozen evidence/protocol changed'
    verify_originals()

def prompt_for(packet):
    return INSTRUCTIONS+'\n\nEVIDENCE JSON:\n'+json.dumps({k:v for k,v in packet.items() if k!='instructions'},separators=(',',':'))

def parse(text,want):
    s=text.strip()
    if s.startswith('```'): s=re.sub(r'^```(?:json)?\s*|\s*```$','',s,flags=re.I)
    obj=json.loads(s); assert isinstance(obj,dict) and set(obj)=={'results'}
    rows=obj['results']; assert isinstance(rows,list) and len(rows)==len(want)
    assert sorted(r['site'] for r in rows)==sorted(want)
    keys={'site','audit_adjustment','leak_probability','verdict','audit_reason'}
    for r in rows:
        assert set(r)==keys
        assert type(r['audit_adjustment']) is int and r['audit_adjustment'] in [-2,-1,0,1,2]
        assert type(r['leak_probability']) in [float,int] and 0<=r['leak_probability']<=1
        assert r['verdict'] in ['LEAK','NO LEAK','AMBIGUOUS']
        assert isinstance(r['audit_reason'],str) and len(r['audit_reason'].split())<=25
    return rows

def codex_call(prompt):
    folder=Path(tempfile.gettempdir())/'paper6_external_codex_blind'; folder.mkdir(exist_ok=True)
    cmd=['codex','exec','--ignore-user-config','--skip-git-repo-check','--ephemeral','--sandbox','read-only','--model','gpt-6.1-sol','--config','model_reasoning_effort=low','--cd',str(folder),'--json','-']
    t=time.monotonic()
    r=subprocess.run(cmd,input=prompt,text=True,encoding='utf-8',capture_output=True,timeout=360)
    events=[json.loads(line) for line in r.stdout.splitlines() if line.strip().startswith('{')]
    tools=[e for e in events if e.get('item',{}).get('type') not in [None,'agent_message','reasoning']]
    messages=[e['item']['text'] for e in events if e.get('type')=='item.completed' and e.get('item',{}).get('type')=='agent_message']
    usage=next((e.get('usage',{}) for e in reversed(events) if e.get('type')=='turn.completed'),{})
    meta=dict(provider='Codex CLI',requested_model='gpt-6.1-sol',returned_model='Not exposed in JSONL; explicit model flag recorded',returncode=r.returncode,event_types=[e.get('type') for e in events],tool_event_count=len(tools),usage=usage,seconds=time.monotonic()-t,stdout_sha256=textsha(r.stdout),stderr_sha256=textsha(r.stderr))
    if r.returncode or tools or not messages:
        meta['failure_events']=events; meta['stderr']=r.stderr[-3000:]
        raise RuntimeError(json.dumps(meta))
    return messages[-1],meta

def poe_call(model,prompt):
    payload=dict(model=model,messages=[dict(role='user',content=prompt)],temperature=0,max_tokens=6000,tools=[],tool_choice='none',extra_body=OPTIONS[model])
    start=time.monotonic()
    for attempt in range(3):
        r=requests.post('https://api.poe.com/v1/chat/completions',headers={'Authorization':'Bearer '+auth()},json=payload,timeout=(30,300))
        if r.status_code in [408,429,500,502,503,529] and attempt<2:
            time.sleep(min(10*(attempt+1),30)); continue
        if not r.ok: raise RuntimeError(f'Provider HTTP {r.status_code}; no credential values logged')
        obj=r.json(); choice=obj['choices'][0]; message=choice['message']
        assert not message.get('tool_calls'),'Provider attempted a tool call'
        return message['content'],dict(provider='Poe',requested_model=model,returned_model=obj.get('model'),response_id=obj.get('id'),usage=obj.get('usage'),finish_reason=choice.get('finish_reason'),seconds=time.monotonic()-start,transport_attempts=attempt+1,raw_response_sha256=hashlib.sha256(r.content).hexdigest(),tool_call_count=0)

def job(model,packet_name):
    dest=OUT/'responses'/model/f'{packet_name}.json'
    packet=load(OUT/'packets'/f'{packet_name}.json'); want=[r['site'] for r in packet['targets']]
    prompt=prompt_for(packet); psha=textsha(prompt)
    if dest.exists():
        cached=load(dest); assert cached['prompt_sha256']==psha; parse(json.dumps({'results':cached['results']}),want)
        return model,packet_name,'cached'
    for attempt in range(2):
        try:
            text,meta=codex_call(prompt) if model=='codex-gpt-6.1-sol' else poe_call(model,prompt)
        except Exception as e:
            save(OUT/'failures'/model/f'{packet_name}_transport_{attempt+1}.json',dict(time_utc=now(),prompt_sha256=psha,error=str(e)[:12000]))
            # A failed provider call is not an alternative scientific response.
            return model,packet_name,'transport_failed'
        try:
            rows=parse(text,want)
        except Exception as e:
            save(OUT/'failures'/model/f'{packet_name}_format_{attempt+1}.json',dict(time_utc=now(),prompt_sha256=psha,error=str(e),raw_text=text,metadata=meta))
            continue
        save(dest,dict(time_utc=now(),model=model,packet=packet_name,prompt_sha256=psha,format_attempt=attempt+1,metadata=meta,results=rows))
        return model,packet_name,'ok'
    return model,packet_name,'format_failed'

def run():
    verify_frozen(); packets=load(OUT/'packet_index.json')
    if not (OUT/'run_start.json').exists(): save(OUT/'run_start.json',dict(started_utc=now(),poe_start_balance=balance(),codex_version=subprocess.run(['codex','--version'],capture_output=True,text=True).stdout.strip()))
    # One packet per model at a time; all three see identical batches, no history.
    for ix,p in enumerate(packets):
        if balance()<15000: say('Existing Poe balance reserve reached; keeping incomplete run without substitutions.'); break
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(job,m,p['packet']) for m in MODELS]
            results=[f.result() for f in as_completed(futures)]
        for result in results: say('RUN',ix+1,'/',len(packets),*result)
        if any(r[2] not in ['ok','cached'] for r in results):
            say('Stopped on failed packet; results preserved. No outcome-based retries.'); break
    status(); verify_originals()

def status():
    packets=load(OUT/'packet_index.json'); expected={x['packet'] for x in packets}
    rows=[]
    for model in MODELS:
        paths=list((OUT/'responses'/model).glob('P*.json'))
        rows.append(dict(model=model,packets=len(paths),expected=len(packets),cases=sum(len(load(f)['results']) for f in paths),complete={p.stem for p in paths}==expected))
    save(OUT/'run_status.json',dict(time_utc=now(),models=rows))
    say(json.dumps(rows))
    return rows

def metrics(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
    return dict(AUC=float(roc_auc_score(y,p)),accuracy=float((tn+tp)/len(y)),balanced_accuracy=float((tp/(tp+fn)+tn/(tn+fp))/2),sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),precision=float(tp/max(1,tp+fp)),F1=float(2*tp/max(1,2*tp+fp+fn)),Brier=float(brier_score_loss(y,p)),TP=int(tp),TN=int(tn),FP=int(fp),FN=int(fn))

def bootstrap(y,probs,groups):
    rng=np.random.default_rng(7103); unique=np.unique(groups); parts={g:np.flatnonzero(groups==g) for g in unique}; result={k:[] for k in probs}
    for _ in range(2000):
        idx=np.concatenate([parts[g] for g in rng.choice(unique,len(unique),replace=True)])
        yy=y[idx]; positive=yy==1; n1=positive.sum(); n0=len(idx)-n1
        if not n1 or not n0: continue
        for model,p in probs.items(): result[model].append(float((rankdata(p[idx])[positive].sum()-n1*(n1+1)/2)/(n1*n0)))
    return {k:np.asarray(v) for k,v in result.items()}

def score():
    verify_frozen(); state=status(); assert all(x['complete'] for x in state),'Do not unblind/score an incomplete model comparison'
    paths=sorted((OUT/'responses').glob('*/*.json'))
    lock=OUT/'locked_responses.json'
    if lock.exists(): assert all(sha(OUT/p)==h for p,h in load(lock)['sha256'].items())
    else: save(lock,dict(locked_utc=now(),sha256={str(p.relative_to(OUT)):sha(p) for p in paths}))
    d=pd.read_csv(OUT/'private_target_index.csv')
    for model in MODELS:
        rows=[r for f in (OUT/'responses'/model).glob('*.json') for r in load(f)['results']]
        assert len(rows)==len(d) and len({r['site'] for r in rows})==len(d)
        mapping={r['site']:r for r in rows}
        d['audit_'+model]=d.site.map(lambda s:mapping[s]['audit_adjustment'])
        d['own_'+model]=d.site.map(lambda s:mapping[s]['leak_probability'])
        d['reason_'+model]=d.site.map(lambda s:mapping[s]['audit_reason'])
        q=np.clip(d.rf_probability.to_numpy(),.02,.98)
        d['primary_'+model]=1/(1+np.exp(-(np.log(q/(1-q))+.7*d['audit_'+model].to_numpy())))
        d['mean_'+model]=(d.rf_probability+d['own_'+model])/2
    d.to_csv(OUT/'scored_predictions.csv',index=False)
    result=[]; paired=[]; behavior=[]
    for dataset,df in d.groupby('dataset'):
        y=df.label.to_numpy(); raw=df.rf_probability.to_numpy()
        probs={'RF tool':raw,'RF zero-audit clipped control':np.clip(raw,.02,.98)}
        for model in MODELS:
            for method in ['primary','own','mean']: probs[f'{model} | {method}']=df[method+'_'+model].to_numpy()
        boots=bootstrap(y,probs,df.group.to_numpy())
        for model,p in probs.items():
            lo,hi=np.percentile(boots[model],[2.5,97.5]); result.append(dict(dataset=dataset,model=model,n=len(y),AUC_cluster_lo=lo,AUC_cluster_hi=hi,**metrics(y,p)))
        for model in MODELS:
            name=model+' | primary'; p=probs[name]
            for ref in ['RF tool','RF zero-audit clipped control']+[m+' | primary' for m in MODELS if m!=model]:
                dif=boots[name]-boots[ref]; lo,hi=np.percentile(dif,[2.5,97.5]); a=(p>=.5)==y; b=(probs[ref]>=.5)==y
                win=int((a&~b).sum()); lose=int((~a&b).sum())
                paired.append(dict(dataset=dataset,model=name,reference=ref,delta_AUC=float(roc_auc_score(y,p)-roc_auc_score(y,probs[ref])),delta_AUC_cluster_lo=lo,delta_AUC_cluster_hi=hi,McNemar_two_sided=float(binomtest(win,win+lose,.5).pvalue) if win+lose else 1.,classification_fixed=win,classification_broken=lose,valid_bootstrap=len(dif)))
            a=df['audit_'+model].to_numpy(); changed=a!=0
            behavior.append(dict(dataset=dataset,model=model,adjusted=int(changed.sum()),downgraded=int((a<0).sum()),downgraded_true_leaks=int(((a<0)&(y==1)).sum()),upgraded=int((a>0).sum()),upgraded_true_leaks=int(((a>0)&(y==1)).sum()),toward_truth=int((changed&(np.abs(p-y)<np.abs(raw-y))).sum()),away_from_truth=int((changed&(np.abs(p-y)>np.abs(raw-y))).sum()),adjustment_counts=json.dumps(dict(Counter(map(str,a))))))
    m=pd.DataFrame(result); p=pd.DataFrame(paired); b=pd.DataFrame(behavior)
    m.to_csv(OUT/'metrics.csv',index=False); p.to_csv(OUT/'paired_comparisons.csv',index=False); b.to_csv(OUT/'audit_behavior.csv',index=False)
    save(OUT/'verification.json',dict(time_utc=now(),original_sources_unchanged=True,original_chatgpt_fingerprints_matched='15/15',frozen_packets_verified=True,response_files=len(paths),cases_per_model=len(d),models_complete=state,labels_not_sent_to_inference=True,no_target_names_or_filenames_sent=True))
    report(m,p,b)
    say(m[(m.model=='RF tool')|m.model.str.endswith(' | primary')][['dataset','model','AUC','balanced_accuracy','sensitivity','specificity','FP','FN']].to_string(index=False))

def report(m,p,b):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    primary=m[(m.model=='RF tool')|m.model.str.endswith(' | primary')].copy()
    primary['model']=primary.model.str.replace(' | primary','',regex=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,(dataset,df) in zip(axes,primary.groupby('dataset')):
        x=np.arange(len(df)); ax.bar(x,df.AUC,color=['#397e9f','#399a7a','#c47b36','#8a6ca5'])
        ax.set(xticks=x,xticklabels=df.model,ylim=(.4,1),ylabel='AUC',title=dataset.title()); ax.tick_params(axis='x',labelrotation=30)
        for i,v in enumerate(df.AUC): ax.text(i,v+.01,f'{v:.3f}',ha='center')
    fig.savefig(OUT/'external_hybrid_comparison.png',dpi=180); plt.close(fig)
    def table(x): return x.to_html(index=False,border=0,float_format=lambda v:f'{v:.4f}')
    paired=p[p.reference=='RF tool'][['dataset','model','delta_AUC','delta_AUC_cluster_lo','delta_AUC_cluster_hi','classification_fixed','classification_broken','McNemar_two_sided']]
    summary=primary[['dataset','model','n','AUC','AUC_cluster_lo','AUC_cluster_hi','accuracy','balanced_accuracy','sensitivity','specificity','FP','FN','Brier']]
    usage=[]
    for model in MODELS:
        calls=[load(f) for f in (OUT/'responses'/model).glob('*.json')]
        usage.append(dict(model=model,calls=len(calls),format_retries=sum(c['format_attempt']-1 for c in calls),tool_events=sum(c['metadata'].get('tool_event_count',c['metadata'].get('tool_call_count',0)) for c in calls)))
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Paper 6 external blind hybrids</title><style>body{{font:16px/1.5 system-ui;max-width:1450px;margin:35px auto;padding:0 24px;color:#20333f}}h1,h2{{color:#195873}}table{{border-collapse:collapse;display:block;overflow:auto;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #cbd7de;text-align:left}}th{{background:#edf3f6}}img{{max-width:100%}}.lead{{padding:18px;background:#edf5fa;border-left:5px solid #246a90}}</style>
<h1>Blind external acoustic hybrid comparison</h1><p>Completed {html.escape(now())}. Separate exploratory external protocol; original Paper 6 predictions preserved.</p>
<p class="lead">All three models audited the same RF tools and the same 1,194 blinded external cases: 964 Dongguan clips and 230 Sheffield channel-records. The OpenAI arm is the user-requested <strong>Codex GPT-6.1-Sol</strong>, not GPT-5.4. Claude Sonnet 5.5 and Gemini 4 Hydrogen ran through Poe. The target is the source recording's leak condition, not a field excavation recommendation.</p>
<h2>What differs between the RF tool and the individual hybrids?</h2><p>The RF tool is the supervised classifier alone. There is one common RF tool score per held-out case; it is not a separate classifier for each LLM. Codex, Claude and Gemini each see that score, its local explanation, training-only reference statistics/reliability and five nearby known training examples. Each independently returns an adjustment from −2 to +2. Its primary hybrid score is <code>sigmoid(logit(clip(p_RF,0.02,0.98))+0.7×adjustment)</code>. An adjustment of +1 approximately doubles the RF odds; −1 approximately halves them. Zero leaves the clipped RF score unchanged. Comparing each hybrid with the RF tool estimates the incremental effect of the LLM audit. Own LLM probabilities and their 50:50 mean with RF are secondary outputs, not the primary hybrid.</p>
<h2>Primary results</h2>{table(summary)}<img src="external_hybrid_comparison.png" alt="External RF and hybrid AUCs">
<p>The probability threshold was fixed at 0.5. Fractional metrics range from 0 to 1. AUC intervals use paired cluster bootstrapping by inferred Dongguan source group or Sheffield experimental condition (2,000 attempted draws, seed 7103). These intervals condition on trained tools and one saved LLM run; they do not include refitting or run-to-run generation uncertainty.</p>
<h2>Increment over the common RF tool</h2>{table(paired)}
<p>Differences are model minus RF. Intervals containing zero do not establish an improvement. McNemar is the exact two-sided test of discordant classifications; all comparisons are exploratory and unadjusted for multiplicity. The full paired table also includes hybrid-versus-hybrid comparisons and a zero-audit clipped-RF control to distinguish clipping from LLM adjustments.</p>
<h2>Audit behavior</h2>{table(b)}
<p>Toward/away counts describe whether a nonzero adjustment reduces/increases absolute error from the observed binary source label. They are descriptive rather than causal evidence of why a leak produces a particular spectrum.</p>
<h2>All secondary scores</h2>{table(m)}
<h2>Fairness and leakage controls</h2><p>The <a href="PROTOCOL.json">protocol</a>, prompts, target IDs, folds and evidence packets were fingerprinted before any model received an external case. Target filenames, dataset names, labels, group IDs and other-model outputs were withheld. Each API request or Codex session was fresh; references and reliability tables were computed from training data only. All three models received byte-identical task text and numerical evidence per packet. No threshold or prompt was adjusted after reviewing external hybrid outcomes. The coordinator already knew the acoustic baseline results and source labels from the earlier analysis, so this is an exploratory model-blinded evaluation, not a claim of a previously unseen or independently preregistered dataset.</p>
<p>Dongguan uses the five original grouped folds, exact-deduplicated signals and training-only nested reliability estimates. Sheffield is strict source-to-target transfer: the RF, retrieval scaler, references, feature explanations and reliability tables use Dongguan alone. Sheffield labels are opened only for scoring after all responses are locked. The full-precision RF predictions reproduce the earlier acoustic baseline; packet RF probabilities are rounded to six decimal places for every model. Primary scoring retains the original full precision.</p>
<p>Codex ran with <code>--model gpt-6.1-sol --ephemeral --ignore-user-config --sandbox read-only</code>, low reasoning effort, and tool events audited. Its backend revision is not exposed in JSONL. Poe requested temperature 0, low provider-specific effort, no tools and web search disabled. Provider wrappers/system prompts and effort controls differ, so equal task evidence does not imply identical inference environments or compute. Format failures could receive one identical-prompt fresh retry, disclosed below; unfavorable valid predictions were never rerun.</p>{table(pd.DataFrame(usage))}
<h2>How this differs from the original Paper 6 H2 run</h2><p>The original task used nine deployment features, an RF with 100 trees, multi-night persistence, spatial peer context, and operational triage. This external task uses 13 amplitude-normalized acoustic descriptors and a 300-tree RF, as in the already-established external baseline. It retains the audit scoring formula, training-only ML/RAG grounding and a shared RF comparator, but omits unavailable evidence and work orders. Its metrics cannot be directly pooled with the original 70-deployment results or described as full operational validation.</p>
<p>Dongguan source groups are inferred from filenames, not proven independent sites. Sheffield has only eight experimental conditions and few independent negative recordings; its 230 channel-records are dependent. A distant sensor recorded while a leak was open still carries the source leak-condition label, which need not mean locally audible leak noise. These distinctions matter when a model treats weak or tonal acoustics as evidence against a leak. Public dataset identity could have been represented in pretraining; anonymizing numerical packets cannot prove otherwise.</p>
<h2>Sources and artifacts</h2><p>Source collections: <a href="https://zenodo.org/records/18631450">Dongguan acoustic dataset</a> and <a href="https://doi.org/10.15131/shef.data.32229270.v2">Sheffield CID laboratory dataset v2</a>, both CC BY 4.0. Existing download checksums, licenses and raw data remain in the earlier external-validation folder. No new raw downloads were required.</p>
<p>Retained: frozen protocol and packets, protected original-source hashes, private scoring index, model predictions and metadata, failed-response audit records if any, <a href="metrics.csv">metrics</a>, <a href="paired_comparisons.csv">paired comparisons</a>, <a href="audit_behavior.csv">audit behavior</a>, <a href="scored_predictions.csv">per-case scores and reasons</a>, and <a href="verification.json">verification</a>. The model outputs were <a href="locked_responses.json">locked before scoring</a>. Generated caches and empty scratch directories are removed after analysis and recorded in cleanup_log.json. Original Paper 6 outputs are hash-checked and never modified.</p></html>'''
    (OUT/'REPORT.html').write_text(body,encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=['prepare','run','status','score']); args=ap.parse_args()
    globals()[args.command]()
