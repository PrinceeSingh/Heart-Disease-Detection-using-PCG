"""
preprocessing.py
Core signal processing pipeline for PCG heart sound recordings.
Identical to the pipeline used in training (Phase 3).
"""
import numpy as np
import librosa
import pywt
from scipy.signal import butter, filtfilt

TARGET_SR = 2000

def bandpass_filter(y, sr=TARGET_SR, low=20, high=400):
    nyq = sr / 2
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, y)

def despike(y, z_thresh=8):
    med = np.median(y)
    mad = np.median(np.abs(y - med)) + 1e-8
    z = np.abs(y - med) / (1.4826 * mad)
    spike_idx = np.where(z > z_thresh)[0]
    y_clean = y.copy()
    for i in spike_idx:
        lo, hi = max(i-2, 0), min(i+3, len(y))
        neighbors = np.delete(y[lo:hi], np.where(np.arange(lo, hi) == i))
        if len(neighbors) > 0:
            y_clean[i] = np.median(neighbors)
    return y_clean

def wavelet_denoise(y, wavelet='db4', level=4):
    coeffs = pywt.wavedec(y, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(y)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[:len(y)]

def normalize_amplitude(y):
    y = y - np.mean(y)
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y

def trim_silence(y, top_db=25):
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed if len(y_trimmed) > 0 else y

def preprocess_clip(path, target_sr=TARGET_SR):
    y, sr = librosa.load(path, sr=None)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    y = bandpass_filter(y, target_sr)
    y = despike(y)
    y = wavelet_denoise(y)
    y = trim_silence(y)
    y = normalize_amplitude(y)
    return y.astype(np.float32), target_sr