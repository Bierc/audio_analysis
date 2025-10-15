from pathlib import Path
import numpy as np
import pandas as pd
import librosa, librosa.display
from scipy.signal import hilbert

SR = 22050
TARGET_DUR = 5.0
N_MFCC = 13
ROLLOFF_PCT = 0.85

def load_pad_mono(path: Path, sr=SR, target_dur=TARGET_DUR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    n = int(target_dur * sr)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    else:
        y = y[:n]
    return y

def rms(x): return float(np.sqrt(np.mean(x**2) + 1e-12))

def match_rms(y, ref_level=None):
    if ref_level is None: ref_level = 10 ** (-20 / 20)
    y_r = rms(y); 
    if y_r < 1e-9: return y
    g = ref_level / y_r
    return np.clip(y * g, -1.0, 1.0)

def envelope(y):
    env = np.abs(hilbert(y))
    return librosa.util.normalize(env)

def attack_time(env, sr=SR, thresh_db=-3):
    env_db = librosa.amplitude_to_db(env + 1e-9, ref=np.max)
    i = np.argmax(env_db >= thresh_db)
    return i / sr

def spectral_block(y, sr=SR):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_pow = S**2
    return {
        "centroid": librosa.feature.spectral_centroid(S=S, sr=sr),
        "bandwidth": librosa.feature.spectral_bandwidth(S=S, sr=sr),
        "rolloff": librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=ROLLOFF_PCT),
        "flatness": librosa.feature.spectral_flatness(S=S),
        "contrast": librosa.feature.spectral_contrast(S=S, sr=sr),
        "flux": librosa.onset.onset_strength(S=librosa.power_to_db(S_pow, ref=np.max))[None, :],
        "zcr": librosa.feature.zero_crossing_rate(y),
    }

def perceptual_block(y, sr=SR, n_mfcc=N_MFCC):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel, ref=np.max), sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_rate = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)) / (len(y)/sr + 1e-9)
    return mfcc, chroma, tempo, onset_rate

def hnr_proxy(y, sr=SR):
    y = y - np.mean(y)
    if np.max(np.abs(y)) > 0: y = y / np.max(np.abs(y))
    ac = librosa.autocorrelate(y); ac = ac / (np.max(ac) + 1e-9)
    minlag, maxlag = int(sr/500), int(sr/50)
    peak = np.max(ac[minlag:maxlag]) if maxlag > minlag else 0
    noise = np.mean(ac[maxlag:]) if len(ac) > maxlag else 1e-9
    return float(10*np.log10((peak+1e-9)/(noise+1e-9)))

def summarize_matrix(name, M):
    M = np.atleast_2d(M)
    return {
        f"{name}_mean": float(np.mean(M)),
        f"{name}_std": float(np.std(M)),
        f"{name}_median": float(np.median(M)),
    }

def extract_one(path: Path, instrument: str, split: str):
    y = load_pad_mono(path)
    y = match_rms(y)
    env = envelope(y)
    spec = spectral_block(y)
    mfcc, chroma, tempo_est, onset_rate = perceptual_block(y)
    feats = {
        "split": split,
        "instrument": instrument,
        "file": path.name,
        "rms": rms(y),
        "attack_time_s": attack_time(env),
        "hnr_proxy_db": hnr_proxy(y),
        "tempo_est_bpm": float(tempo_est),
        "onset_rate_s": float(onset_rate),
    }
    for k in ["centroid","bandwidth","rolloff","flatness","flux","zcr","contrast"]:
        feats.update(summarize_matrix(k, spec[k]))
    for i in range(mfcc.shape[0]):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))
    for i in range(12):
        feats[f"chroma{i}_mean"] = float(np.mean(chroma[i]))
    return feats

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    manifest = pd.read_csv(ROOT/"outputs/manifest.csv")
    rows = []
    for _, r in manifest.iterrows():
        p = Path(r["path"])
        feats = extract_one(p, instrument=r["instrument"], split=r["split"])
        rows.append(feats)
    df = pd.DataFrame(rows)
    out_single = ROOT/"outputs/features_single.csv"
    out_melo   = ROOT/"outputs/features_melodies.csv"
    df[df["split"]=="single"].drop(columns=["split"]).to_csv(out_single, index=False)
    df[df["split"]=="melody"].drop(columns=["split"]).to_csv(out_melo, index=False)
    print(f"Saved:\n- {out_single}\n- {out_melo}")
