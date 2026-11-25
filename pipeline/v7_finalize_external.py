"""Verify the completed external experiment, annotate its report, and clean caches."""
from pathlib import Path
import hashlib, html, json, tempfile
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'derived/v7_external_hybrids'
MODELS=['codex-gpt-6.1-sol','claude-sonnet-5.5','gemini-4-hydrogen']
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    frozen=load(OUT/'frozen_manifest.json')
    assert sha(ROOT/'code/v7_external_hybrids.py')==frozen['scoring_script_sha256']
    assert all(sha(OUT/p)==h for p,h in frozen['sha256'].items())
    originals=load(OUT/'protected_sources.json'); assert all(sha(Path(p))==h for p,h in originals.items())
    locks=load(OUT/'locked_responses.json'); assert all(sha(OUT/p)==h for p,h in locks['sha256'].items())
    packets=load(OUT/'packet_index.json'); assert len(packets)==53
    tool_count=0; totals=[]
    for packet in packets:
        docs=[load(OUT/'responses'/m/(packet['packet']+'.json')) for m in MODELS]
        assert len({x['prompt_sha256'] for x in docs})==1
        assert all({r['site'] for r in x['results']}==set(packet['sites']) for x in docs)
        for model,doc in zip(MODELS,docs):
            meta=doc['metadata']; tool_count+=meta.get('tool_event_count',meta.get('tool_call_count',0))
            if model!='codex-gpt-6.1-sol': assert meta['returned_model']==model
    assert tool_count==0
    for model in MODELS:
        docs=[load(f) for f in (OUT/'responses'/model).glob('*.json')]
        assert sum(len(d['results']) for d in docs)==1194
        totals.append(dict(model=model,packets=len(docs),cases=1194,format_retries=sum(d['format_attempt']-1 for d in docs),seconds_summed=sum(d['metadata']['seconds'] for d in docs),input_tokens=sum((d['metadata'].get('usage') or {}).get('input_tokens',(d['metadata'].get('usage') or {}).get('prompt_tokens',0)) for d in docs),output_tokens=sum((d['metadata'].get('usage') or {}).get('output_tokens',(d['metadata'].get('usage') or {}).get('completion_tokens',0)) for d in docs)))
    usage=pd.DataFrame(totals).rename(columns={'input_tokens':'successful_input_tokens','output_tokens':'successful_output_tokens'})
    for model in MODELS:
        failures=[load(f).get('metadata',{}).get('usage') or {} for f in (OUT/'failures'/model).glob('*.json')]
        usage.loc[usage.model==model,'failed_input_tokens']=sum(x.get('input_tokens',x.get('prompt_tokens',0)) for x in failures)
        usage.loc[usage.model==model,'failed_output_tokens']=sum(x.get('output_tokens',x.get('completion_tokens',0)) for x in failures)
    usage.to_csv(OUT/'inference_usage.csv',index=False)
    m=pd.read_csv(OUT/'metrics.csv'); p=pd.read_csv(OUT/'paired_comparisons.csv'); behavior=pd.read_csv(OUT/'audit_behavior.csv')
    text=[]
    for dataset in ['dongguan','sheffield']:
        rows=m[(m.dataset==dataset)&m.model.str.endswith(' | primary')]
        baseline=m[(m.dataset==dataset)&(m.model=='RF tool')].iloc[0]
        text.append(f'<h3>{dataset.title()}</h3><p>Common RF tool: AUC {baseline.AUC:.4f}, balanced accuracy {100*baseline.balanced_accuracy:.1f}%, {int(baseline.FP)} false positives and {int(baseline.FN)} false negatives.</p><ul>')
        for _,r in rows.iterrows():
            ref=p[(p.dataset==dataset)&(p.model==r.model)&(p.reference=='RF tool')].iloc[0]
            interpretation='The paired interval includes zero; improvement is not established.' if ref.delta_AUC_cluster_lo<=0<=ref.delta_AUC_cluster_hi else ('The exploratory paired interval is above zero.' if ref.delta_AUC_cluster_lo>0 else 'The exploratory paired interval is below zero.')
            text.append(f'<li><strong>{html.escape(r.model.replace(" | primary",""))}</strong>: AUC {r.AUC:.4f}, balanced accuracy {100*r.balanced_accuracy:.1f}%, {int(r.FP)} false positives, {int(r.FN)} missed leak-condition channel-records/clips. AUC difference from RF {ref.delta_AUC:+.4f} (95% cluster interval {ref.delta_AUC_cluster_lo:+.4f} to {ref.delta_AUC_cluster_hi:+.4f}). {interpretation}</li>')
        text.append('</ul>')
    summary='<section id="findings"><h2>Findings from the completed blind run</h2>'+''.join(text)+'</section>'
    report=OUT/'REPORT.html'; body=report.read_text(encoding='utf-8')
    body=body.replace('<h2>Primary results</h2>',summary+'<h2>Primary results</h2>') if 'id="findings"' not in body else body
    body=body.replace('byte-identical task text and numerical evidence','the same task text and numerical evidence')
    body=body.replace('Sheffield labels are opened only for scoring after all responses are locked.','Sheffield labels are excluded from all inference packets and used to score the locked responses. The coordinator already knew source labels from the earlier baseline analysis, as disclosed above.')
    execution=load(OUT/'execution_amendment.json')
    note='<h2>Execution and verification</h2><p>Inference scheduling increased to two independent requests per model at a time after packet P005 was fully saved. This throughput-only change did not alter the frozen protocol, scoring code, prompts or evidence; no response was discarded. See execution_amendment.json. All 159 successful packet responses have matched prompt fingerprints across models and zero recorded tool calls. Both Poe model identifiers match the requested identifiers. All 1,194 cases are present for every model, and every original protected source hash is unchanged.</p>'
    if '<h2>Execution and verification</h2>' not in body: body=body.replace('<h2>Sources and artifacts</h2>',note+'<h2>Sources and artifacts</h2>')
    report.write_text(body,encoding='utf-8')
    # Update navigation while preserving the older reports as historical results.
    project_note=ROOT/'PROJECT_NOTE.md'; old=project_note.read_text(encoding='utf-8')
    marker='> **External hybrids completed, 24 November 2026:**'
    if marker not in old:
        annotation=marker+' Codex GPT-6.1-Sol, Claude Sonnet 5.5 and Gemini 4 Hydrogen have now completed the separate acoustic-only blind comparison on all 964 Dongguan and 230 Sheffield cases. See [the external hybrid report](derived/v7_external_hybrids/REPORT.html). The earlier statement below that external LLM runs did not exist describes the pre-follow-up state. Original Paper 6 predictions are unchanged.\n\n'
        pos=old.find('\n')+1; project_note.write_text(old[:pos]+'\n'+annotation+old[pos:],encoding='utf-8')
    earlier=ROOT/'Paper6_Model_Comparison.html'
    if earlier.exists():
        old=earlier.read_text(encoding='utf-8')
        if 'id="external-followup"' not in old:
            pos=old.find('<h1>'); banner='<p class="lead" id="external-followup"><strong>Later follow-up, 24 November 2026:</strong> the three external hybrids have now been run. <a href="derived/v7_external_hybrids/REPORT.html">Open their blind Dongguan/Sheffield comparison</a>. The external-availability table below records the earlier pre-follow-up stage.</p>'
            earlier.write_text(old[:pos]+banner+old[pos:],encoding='utf-8')
    deleted=[]
    candidates=[OUT/'pause_after_packet.py']
    for prefix in ['v7_external_hybrids','v7_external_run_parallel','llm_clients']:
        candidates += list((ROOT/'code/__pycache__').glob(prefix+'.*.pyc'))
    for file in candidates:
        if file.exists():
            assert file.resolve().is_relative_to(ROOT.resolve())
            deleted.append(dict(file=str(file.relative_to(ROOT)),bytes=file.stat().st_size,reason='Generated cache or one-time inference handoff helper'))
            file.unlink()
    cache=ROOT/'code/__pycache__'
    if cache.exists() and not any(cache.iterdir()):
        try: cache.rmdir()
        except PermissionError:
            # Synced folders can retain the read-only directory attribute after file removal.
            import stat
            cache.chmod(stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
            cache.rmdir()
    scratch=(Path(tempfile.gettempdir())/'paper6_external_codex_blind').resolve()
    if scratch.exists() and not any(scratch.iterdir()):
        assert scratch.parent==Path(tempfile.gettempdir()).resolve()
        scratch.rmdir(); deleted.append(dict(file=str(scratch),bytes=0,reason='Empty task-specific Codex scratch directory'))
    log=OUT/'cleanup_log.json'; previous=load(log) if log.exists() else []
    save(log,previous+deleted)
    verification=load(OUT/'verification.json')
    verification.pop('labels_not_sent_to_inference',None)
    verification.update(target_labels_not_sent_to_inference=True,labeled_training_references_intentionally_supplied=True,protected_source_count=len(originals),all_159_response_files_hash_verified=True,all_53_prompt_fingerprints_matched_across_models=True,total_recorded_tool_calls=0,poe_returned_model_ids_verified=True,scoring_code_unchanged_since_freeze=True,failed_response_sha256={str(f.relative_to(OUT)):sha(f) for f in (OUT/'failures').glob('**/*.json')},finalizer_sha256=sha(Path(__file__)))
    save(OUT/'verification.json',verification)
    assert all(sha(Path(p))==h for p,h in originals.items())
    print('Verified 159 responses / 3582 case predictions; original sources unchanged.')
    print('Deleted',len(deleted),'unnecessary items;',sum(x['bytes'] for x in deleted),'bytes.')

if __name__=='__main__': main()
