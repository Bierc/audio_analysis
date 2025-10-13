from pathlib import Path
import numpy as np
import pandas as pd
import librosa, librosa.display
import matplotlib.pyplot as plt
from scipy.signal import hilbert


# Audio configuration
SR = 22050
TARGET_DUR = 5.0
N_MFCC = 13
ROLLOFF_PCT = 0.85

def load_pad_mono(path: Path, sr: int = SR, target_dur: float = TARGET_DUR):
    """Load audio, convert to mono, and pad or truncate to target duration."""
    y, _sr = librosa.load(path, sr=sr, mono=True)
    target_n = int(target_dur * sr)
    if len(y) < target_n:
        y = np.pad(y, (0, target_n - len(y)))
    else:
        y = y[:target_n]
    return y


def rms(x):
    """Root Mean Square — proxy for loudness."""
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def match_rms(y, ref_level: float = None):
    """Normalize the audio RMS to a reference level (-20 dBFS default)."""
    if ref_level is None:
        ref_level = 10 ** (-20 / 20)
    y_rms = rms(y)
    if y_rms < 1e-9:
        return y
    gain = ref_level / y_rms
    return np.clip(y * gain, -1.0, 1.0)


def amplitude_envelope(y):
    """Estimate amplitude envelope via analytic signal (Hilbert transform)."""
    analytic = hilbert(y)
    env = np.abs(analytic)
    return librosa.util.normalize(env)


def estimate_attack_time(env, sr=SR, thresh_db=-3):
    """Estimate time (in s) to reach -3 dB below peak amplitude."""
    env_db = librosa.amplitude_to_db(env + 1e-9, ref=np.max)
    idx = np.argmax(env_db >= thresh_db)
    return idx / sr


def spectral_block(y, sr=SR):
    """Extract core spectral features."""
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_power = S**2
    feats = {
        "centroid": librosa.feature.spectral_centroid(S=S, sr=sr),
        "bandwidth": librosa.feature.spectral_bandwidth(S=S, sr=sr),
        "rolloff": librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=ROLLOFF_PCT),
        "flatness": librosa.feature.spectral_flatness(S=S),
        "contrast": librosa.feature.spectral_contrast(S=S, sr=sr),
        "flux": librosa.onset.onset_strength(S=librosa.power_to_db(S_power, ref=np.max))[None, :],
        "zcr": librosa.feature.zero_crossing_rate(y),
    }
    return feats


def perceptual_block(y, sr=SR, n_mfcc=N_MFCC):
    """Extract perceptual features: MFCCs, chroma, tempo, onset rate."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel, ref=np.max), sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_rate = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)) / (len(y) / sr + 1e-9)
    return mfcc, chroma, tempo, onset_rate


def hnr_proxy(y, sr=SR):
    """Approximate Harmonic-to-Noise Ratio (HNR) via autocorrelation."""
    y = y - np.mean(y)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    ac = librosa.autocorrelate(y)
    ac = ac / (np.max(ac) + 1e-9)
    minlag = int(sr / 500)
    maxlag = int(sr / 50)
    peak = np.max(ac[minlag:maxlag]) if maxlag > minlag else 0
    noise = np.mean(ac[maxlag:]) if len(ac) > maxlag else 1e-9
    return float(10 * np.log10((peak + 1e-9) / (noise + 1e-9)))


def summarize_matrix(name, M):
    """Return mean, std, and median for a given feature matrix."""
    M = np.atleast_2d(M)
    return {
        f"{name}_mean": float(np.mean(M)),
        f"{name}_std": float(np.std(M)),
        f"{name}_median": float(np.median(M)),
    }
