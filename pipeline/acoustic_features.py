"""Recording- and deployment-level acoustic features used by the evidence tools (Section 2.2 of the paper).

Recording features follow Welch's method (1,024-sample Hann segments, 50% overlap) on de-meaned audio sampled at 4,096 Hz.
Deployment features aggregate recordings of one logger at one location; persistence uses the nightly median leak-band ratio.
"""
import numpy as np, scipy.signal as signal

LEAK_THR = 0.35   # set during exploratory analysis; not tuned inside validation folds


def recording_features(audio, sr=4096):
    a = audio.astype(np.float32); a = a - a.mean()
    rms = np.sqrt(np.mean(a ** 2)) + 1e-9
    env = np.abs(signal.hilbert(a)); env_var = np.var(env) / ((np.mean(env) + 1e-9) ** 2)
    nper = min(1024, len(a)); fr, psd = signal.welch(a, fs=sr, nperseg=nper, noverlap=nper // 2); tot = np.sum(psd) + 1e-12
    def band(lo, hi, incl=False):
        m = (fr >= lo) & ((fr <= hi) if incl else (fr < hi)); return np.sum(psd[m])
    e = [band(0, 200), band(200, 600), band(600, 1000), band(1000, 1500), band(1500, 2048, True)]
    cen = float(np.sum(fr * psd) / tot)
    return dict(rms_amplitude=float(rms), envelope_variance=float(env_var), leak_band_ratio=float((e[1] + e[2]) / tot),
                band_fractions=[float(x / tot) for x in e], spectral_centroid=cen,
                spectral_spread=float(np.sqrt(np.sum(((fr - cen) ** 2) * psd) / tot)),
                spectral_flatness=float(np.exp(np.mean(np.log(psd + 1e-12))) / (np.mean(psd) + 1e-12)))


def deployment_features(recs):
    """recs: pandas DataFrame of one deployment with columns rec_date, leak_band_ratio, rms_amplitude, spectral_flatness,
    spectral_centroid, envelope_variance."""
    nights = recs.groupby("rec_date")["leak_band_ratio"].median(); rms = recs.rms_amplitude.values
    return dict(n_nights=int(len(nights)), mean_rms=float(rms.mean()), median_rms=float(np.median(rms)),
                mean_leak_ratio=float(recs.leak_band_ratio.mean()), persistence_ratio=float((nights >= LEAK_THR).mean()),
                transient_ratio=float(np.percentile(rms, 95) / (np.median(rms) + 1e-9)),
                stationarity_index=float(recs.leak_band_ratio.mean() / (recs.leak_band_ratio.std(ddof=0) + 1e-9)),
                mean_flatness=float(recs.spectral_flatness.mean()), mean_centroid=float(recs.spectral_centroid.mean()),
                mean_env_var=float(recs.envelope_variance.median()))
