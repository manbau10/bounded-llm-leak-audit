"""Fixed acoustic baseline and independent transfer test; no blind inputs read."""
from pathlib import Path
import hashlib, json, re, sys, platform
import numpy as np
import pandas as pd
import scipy, scipy.signal as sig
import soundfile as sf
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FEATURES = ['centroid','bandwidth','flatness','entropy','peak_fraction','zcr','kurtosis'] + [f'band_{a}_{b}' for a,b in [(20,100),(100,250),(250,500),(500,1000),(1000,2000),(2000,3900)]]

def features(x, fs):
    x=np.asarray(x,dtype=float); x=x-x.mean()
    if fs != 8000:
        x=sig.resample_poly(x,8000,round(fs))
    rms=np.sqrt(np.mean(x*x)); z=x/max(rms,1e-15)
    f,p=sig.welch(z,8000,nperseg=min(1024,len(z)))
    use=(f>=20)&(f<=3900); f=f[use]; p=p[use]+1e-30; q=p/p.sum()
    center=np.sum(f*q)
    v=[center/4000,np.sqrt(np.sum((f-center)**2*q))/4000,np.exp(np.mean(np.log(p)))/np.mean(p),-np.sum(q*np.log(q))/np.log(len(q)),q.max(),np.mean(z[1:]*z[:-1]<0),np.mean(z**4)]
    v += [q[(f>=a)&(f<b)].sum() for a,b in [(20,100),(100,250),(250,500),(500,1000),(1000,2000),(2000,3900)]]
    return v

def model():
    return RandomForestClassifier(n_estimators=300,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=-1)

def metrics(y,p):
    y=np.asarray(y); p=np.asarray(p); pred=p>=.5
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(n=len(y),n_leak=int(y.sum()),n_no_leak=int((y==0).sum()),auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,balanced_accuracy=float(balanced_accuracy_score(y,pred)) if len(set(y))==2 else None,sensitivity=float(tp/(tp+fn)) if tp+fn else None,specificity=float(tn/(tn+fp)) if tn+fp else None,brier=float(brier_score_loss(y,p)),confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]])

def ci(df,group):
    rng=np.random.default_rng(42); groups=df[group].unique(); samples=[]
    parts={g:df[df[group]==g] for g in groups}
    for _ in range(1000):
        b=pd.concat([parts[g] for g in rng.choice(groups,len(groups),replace=True)])
        if b.label.nunique()==2:
            samples.append([roc_auc_score(b.label,b.probability),balanced_accuracy_score(b.label,b.probability>=.5)])
    return dict(bootstrap_unit=group,valid_resamples=len(samples),auc_95_percentile=np.quantile(np.array(samples)[:,0],[.025,.975]).tolist(),balanced_accuracy_95_percentile=np.quantile(np.array(samples)[:,1],[.025,.975]).tolist())

def dongguan():
    rows=[]
    for path in sorted((ROOT/'data/dongguan').rglob('*.wav')):
        x,fs=sf.read(path); assert fs==8000 and len(x)==8000 and x.ndim==1
        category='noise' if 'environmental noise' in str(path) else ('no_leak' if 'no leak acoustic data' in str(path) else 'leak')
        group=re.sub(r'[-_]\d+-\d+(?:_\d+)?$','',path.stem)
        rows.append(dict(file=str(path.relative_to(ROOT/'data')),label=int(category=='leak'),category=category,group=group,sha256=hashlib.sha256(x.tobytes()).hexdigest(),**dict(zip(FEATURES,features(x,fs)))))
    df=pd.DataFrame(rows)
    # Unite recording-name groups when any identical decoded waveform links them.
    parents={g:g for g in df.group}
    def find(g):
        while parents[g]!=g: parents[g]=parents[parents[g]]; g=parents[g]
        return g
    for _,d in df.groupby('sha256'):
        gs=list(d.group)
        for g in gs[1:]: parents[find(g)]=find(gs[0])
    df['group']=df.group.map(find)
    conflicts=int((df.groupby('sha256').label.nunique()>1).sum())
    assert conflicts==0, 'Conflicting exact duplicates need explicit review'
    duplicates=int(df.duplicated('sha256').sum()); df=df.drop_duplicates('sha256').copy().reset_index(drop=True)
    df['fold']=-1; df['probability']=np.nan
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    folds=[]
    for fold,(tr,te) in enumerate(cv.split(df[FEATURES],df.label,df.group)):
        assert not set(df.iloc[tr].group)&set(df.iloc[te].group)
        assert not set(df.iloc[tr].sha256)&set(df.iloc[te].sha256)
        m=model().fit(df.iloc[tr][FEATURES],df.iloc[tr].label)
        df.loc[te,'probability']=m.predict_proba(df.iloc[te][FEATURES])[:,1]; df.loc[te,'fold']=fold
        folds.append(dict(fold=fold,train_n=len(tr),test_n=len(te),train_groups=df.iloc[tr].group.nunique(),test_groups=df.iloc[te].group.nunique()))
    df.to_csv(ROOT/'dongguan_predictions_features.csv',index=False)
    result=dict(raw_files=len(rows),exact_duplicate_files_removed=duplicates,unique_groups=df.group.nunique(),overall=metrics(df.label,df.probability),uncertainty=ci(df,'group'),folds=folds,by_category={c:metrics(d.label,d.probability) for c,d in df.groupby('category')})
    # Extra discrimination estimate omits environmental sound; predictions remain unchanged.
    d=df[df.category!='noise']; result['field_clips_only']=metrics(d.label,d.probability)
    return model().fit(df[FEATURES],df.label),df,result

def sheffield(m):
    rows=[]; sources=[]
    for path in sorted((ROOT/'data/sheffield').rglob('*.csv')):
        if 'All_Data' not in path.parts: continue
        d=pd.read_csv(path); t=d.iloc[:,0].to_numpy(); fs=int(round(1/np.median(np.diff(t))))
        assert fs==8192 and np.isfinite(d.to_numpy()).all()
        sources.append(dict(file=str(path.relative_to(ROOT/'data')),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(d),fs=fs,columns=list(d.columns)))
        for col in d.columns[1:]:
            x=d[col].to_numpy(); windows=[features(x[start:start+fs],fs) for start in range(0,len(x)-fs+1,fs)]
            v=np.mean(windows,axis=0)
            rows.append(dict(file=str(path.relative_to(ROOT/'data')),condition=path.parent.name,channel=col,label=int('noleak' not in path.name.lower()),n_windows=len(windows),sha256=hashlib.sha256(x.tobytes()).hexdigest(),**dict(zip(FEATURES,v))))
        print('processed',path.parent.name,path.name,flush=True)
    df=pd.DataFrame(rows); dup=int(df.duplicated('sha256').sum())
    assert (df.groupby('sha256').label.nunique()<=1).all()
    df=df.drop_duplicates('sha256').copy(); df['probability']=m.predict_proba(df[FEATURES])[:,1]
    df.to_csv(ROOT/'sheffield_transfer_predictions_features.csv',index=False)
    (ROOT/'sheffield_record_manifest.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    near=df[df.channel=='Acc0']
    result=dict(source_files=len(sources),raw_channel_records=len(rows),exact_duplicate_channels_removed=dup,unique_channel_records=len(df),overall=metrics(df.label,df.probability),uncertainty=ci(df,'condition'),near_leak_Acc0=metrics(near.label,near.probability),by_condition={c:metrics(d.label,d.probability) for c,d in df.groupby('condition')})
    return df,result

def spatial():
    rows=[]
    for path in sorted((ROOT/'data/sheffield').rglob('*.csv')):
        if 'Coherence (simultaneous recording)' not in path.parts: continue
        d=pd.read_csv(path); family=path.parent.parent.name; distance=int(path.parent.name.split('#')[1])
        a,b=(d.iloc[:,2].values,d.iloc[:,3].values) if '3mmValve' in family else (d.iloc[:,1].values,d.iloc[:,2].values)
        f,c=sig.coherence(a,b,fs=8192,nperseg=2048,noverlap=1024)
        band=(f>=100)&(f<=1000)
        rows.append(dict(file=str(path.relative_to(ROOT/'data')),family=family,distance_m=distance,mean_coherence_100_1000=float(c[band].mean()),fraction_coherence_above_05=float((c[band]>.5).mean())))
    df=pd.DataFrame(rows); df.to_csv(ROOT/'sheffield_spatial_coherence.csv',index=False)
    return df

def main():
    m,dg,r1=dongguan(); sh,r2=sheffield(m); sp=spatial()
    result=dict(protocol='Acoustic component validation, not a rerun or accuracy estimate of the blind agentic audit.',seed=42,features=FEATURES,dongguan=r1,sheffield_zero_shot=r2,versions={k:v for k,v in [('python',sys.version),('numpy',np.__version__),('scipy',scipy.__version__),('pandas',pd.__version__),('sklearn',sklearn.__version__),('soundfile',sf.__version__)]})
    (ROOT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for name,df,color in [('Dongguan grouped CV',dg,'#196a9c'),('Sheffield transfer',sh,'#c36b25')]:
        from sklearn.metrics import RocCurveDisplay
        RocCurveDisplay.from_predictions(df.label,df.probability,name=name,ax=axes[0],color=color)
    axes[0].plot([0,1],[0,1],'k--',lw=.8); axes[0].set_title('Acoustic baseline discrimination')
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay(confusion_matrix(sh.label,sh.probability>=.5),display_labels=['No leak','Leak']).plot(ax=axes[1],colorbar=False,cmap='Blues'); axes[1].set_title('Sheffield: fixed threshold 0.5')
    for family,d in sp.groupby('family'):
        s=d.groupby('distance_m').mean(numeric_only=True)
        axes[2].plot(s.index,s.mean_coherence_100_1000,'o-',label=family)
    axes[2].set(xlabel='Sensor separation (m)',ylabel='Mean coherence, 100–1000 Hz',ylim=(0,1),title='Simultaneous Sheffield recordings'); axes[2].legend(fontsize=8)
    fig.savefig(ROOT/'validation_summary.png',dpi=180); plt.close(fig)
    print(json.dumps({'dongguan':r1['overall'],'sheffield':r2['overall']},indent=2))

if __name__=='__main__': main()
