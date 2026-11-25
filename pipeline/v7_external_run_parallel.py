"""Throughput-only runner for already frozen v7 packets and prediction rules."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import v7_external_hybrids as experiment

def main():
    e=experiment; e.verify_frozen()
    amendment=e.load(e.OUT/'execution_amendment.json')
    assert amendment['transport_requests_intentionally_interrupted']==0
    frozen=e.load(e.OUT/'frozen_manifest.json')
    assert e.sha(e.Path(e.__file__))==frozen['scoring_script_sha256']
    limit={m:threading.Semaphore(2) for m in e.MODELS}; stop=threading.Event()
    jobs=[(m,p['packet']) for p in e.load(e.OUT/'packet_index.json') for m in e.MODELS if not (e.OUT/'responses'/m/(p['packet']+'.json')).exists()]
    amendment.update(parallel_runner_sha256=e.sha(e.Path(__file__)),remaining_jobs_at_restart=len(jobs),start_poe_balance=e.balance(),restart_utc=e.now())
    e.save(e.OUT/'execution_amendment.json',amendment)
    def task(model,packet):
        with limit[model]:
            if stop.is_set(): return model,packet,'not_submitted_after_failure'
            if model!='codex-gpt-6.1-sol' and e.balance()<15000:
                stop.set(); return model,packet,'not_submitted_balance_reserve'
            result=e.job(model,packet)
            if result[2] not in ['ok','cached']: stop.set()
            return result
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(task,*job) for job in jobs]
        for n,future in enumerate(as_completed(futures),1):
            result=future.result(); e.say('PARALLEL',n,'/',len(jobs),*result)
    e.status(); e.verify_originals()
    amendment.update(finished_utc=e.now(),end_poe_balance=e.balance())
    e.save(e.OUT/'execution_amendment.json',amendment)

if __name__=='__main__': main()
