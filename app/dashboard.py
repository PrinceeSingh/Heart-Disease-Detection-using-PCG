import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import pywt
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import butter, filtfilt
from PIL import Image
import tempfile, os, io

# ── page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="PCG Heart Disease Detector",
    page_icon="🫀",
    layout="wide"
)

# ── constants ─────────────────────────────────────────────────
TARGET_SR   = 2000
N_FFT       = 256
HOP_LENGTH  = 32
N_MELS      = 64
WINDOW_SEC  = 4
WINDOW_LEN  = int(WINDOW_SEC * TARGET_SR)

MODEL_PATH  = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pt')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'test_results.json')

# ── load config ───────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    config = json.load(f)
THRESHOLD = config['optimal_threshold']

# ── model definition (must match training exactly) ────────────
class PCGClassifier(nn.Module):
    def __init__(self, n_mels=64, n_frames=251, dropout=0.4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((16, 1)),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.attention  = nn.Linear(128, 1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.lstm(x)
        attn_w = torch.softmax(self.attention(x), dim=1)
        x = (x * attn_w).sum(dim=1)
        return self.classifier(x).squeeze(1)

# ── load model (cached so it only loads once) ─────────────────
@st.cache_resource
def load_model():
    m = PCGClassifier()
    m.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    m.eval()
    return m

model = load_model()

# ── preprocessing (identical to Phase 3 pipeline) ────────────
def bandpass_filter(y, sr, low=20, high=400):
    nyq = sr / 2
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, y)

def despike(y, z_thresh=8):
    med = np.median(y)
    mad = np.median(np.abs(y - med)) + 1e-8
    z   = np.abs(y - med) / (1.4826 * mad)
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
    sigma  = np.median(np.abs(coeffs[-1])) / 0.6745
    thr    = sigma * np.sqrt(2 * np.log(len(y)))
    coeffs[1:] = [pywt.threshold(c, thr, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[:len(y)]

def normalize_amplitude(y):
    y = y - np.mean(y)
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y

def preprocess(y, sr):
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    y = bandpass_filter(y, TARGET_SR)
    y = despike(y)
    y = wavelet_denoise(y)
    y, _ = librosa.effects.trim(y, top_db=25)
    y = normalize_amplitude(y)
    return y.astype(np.float32)

def extract_logmel(window):
    mel = librosa.feature.melspectrogram(
        y=window, sr=TARGET_SR, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
        fmin=20, fmax=400
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

def segment(y):
    if len(y) < WINDOW_LEN:
        return [np.pad(y, (0, WINDOW_LEN - len(y)))]
    windows, start = [], 0
    while start + WINDOW_LEN <= len(y):
        windows.append(y[start:start + WINDOW_LEN])
        start += WINDOW_LEN // 2
    return windows

def signal_quality(y):
    rms  = np.sqrt(np.mean(y**2))
    peak = np.max(np.abs(y))
    cf   = peak / (rms + 1e-8)
    # heuristic: crest factor 5–40 = usable PCG range
    if cf < 3:
        return "poor", cf
    if cf > 60:
        return "poor", cf
    return "good", cf

# ── Grad-CAM ─────────────────────────────────────────────────
def compute_gradcam(x_tensor):
    activations, gradients = {}, {}

    def fwd_hook(m, inp, out):
        activations['v'] = out.detach()
    def bwd_hook(m, gi, go):
        gradients['v'] = go[0].detach()

    handle_f = model.cnn[8].register_forward_hook(fwd_hook)
    handle_b = model.cnn[8].register_backward_hook(bwd_hook)

    x = x_tensor.clone().requires_grad_(True)
    logit = model(x)
    model.zero_grad()
    logit.backward()

    handle_f.remove()
    handle_b.remove()

    weights = gradients['v'].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations['v']).sum(dim=1, keepdim=True))
    cam = cam.squeeze().cpu().numpy()

    cam = np.array(Image.fromarray(cam).resize((251, 64), Image.BILINEAR))
    mn, mx = cam.min(), cam.max()
    return (cam - mn) / (mx - mn + 1e-8)

def make_overlay_figure(logmel, cam, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    extent = [0, 4, 20, 400]
    kw = dict(aspect='auto', origin='lower', extent=extent)

    axes[0].imshow(logmel, cmap='magma', **kw)
    axes[0].set_title('Log-mel Spectrogram')
    axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Frequency (Hz)')

    axes[1].imshow(cam, cmap='jet', vmin=0, vmax=1, **kw)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].set_xlabel('Time (s)')

    axes[2].imshow(logmel, cmap='magma', alpha=0.55, **kw)
    axes[2].imshow(cam,    cmap='jet',   alpha=0.45, vmin=0, vmax=1, **kw)
    axes[2].set_title('Overlay')
    axes[2].set_xlabel('Time (s)')

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

# ── UI ────────────────────────────────────────────────────────
st.title("🫀 Heart Disease Detection via Phonocardiogram")
st.markdown(
    "Upload a PCG heart sound recording (`.wav`). "
    "The system preprocesses the audio, runs the trained CNN-BiLSTM model, "
    "and explains its prediction using Grad-CAM."
)
st.caption(
    "⚠️ Research prototype only — not a certified medical device. "
    "Results must not be used for clinical diagnosis."
)
st.divider()

uploaded = st.file_uploader("Upload a PCG recording (.wav)", type=["wav"])

if uploaded:
    # save to temp file so librosa can read it
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Loading and preprocessing audio..."):
        y_raw, sr = librosa.load(tmp_path, sr=None)
        os.unlink(tmp_path)

        quality_label, cf = signal_quality(y_raw)
        y_clean = preprocess(y_raw, sr)
        windows = segment(y_clean)

    # ── signal quality gate ───────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Duration", f"{len(y_raw)/sr:.1f}s")
    col2.metric("Windows analysed", len(windows))
    col3.metric("Signal quality", quality_label.upper(),
                delta="usable" if quality_label == "good" else "may affect accuracy",
                delta_color="normal" if quality_label == "good" else "inverse")

    if quality_label == "poor":
        st.warning(
            "⚠️ Signal quality check failed (crest factor outside normal PCG range). "
            "The recording may be too noisy or contain mostly silence. "
            "Results may be unreliable — try repositioning the sensor."
        )

    st.divider()

    # ── run inference on all windows, majority vote ───────────
    with st.spinner("Running model inference..."):
        all_probs, all_cams, all_logmels = [], [], []
        for w in windows:
            lm = extract_logmel(w)
            x_t = torch.tensor(lm).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                prob = torch.sigmoid(model(x_t)).item()
            cam = compute_gradcam(x_t)
            all_probs.append(prob)
            all_cams.append(cam)
            all_logmels.append(lm)

            mean_prob = np.mean(all_probs)
            max_prob  = np.max(all_probs)

            # use max for detection, mean for display
            detection_prob = max_prob
            mean_p = mean_prob

            if detection_prob >= THRESHOLD:
                prediction = "ABNORMAL"
                result_color = "error"
            elif mean_p >= 0.4:
                prediction = "BORDERLINE"
                result_color = "warning"
            else:
                prediction = "NORMAL"
                result_color = "success"

            confidence = detection_prob if prediction == "ABNORMAL" else (
                1 - mean_p if prediction == "NORMAL" else
                abs(mean_p - 0.5) * 2
            )

    # ── result banner ─────────────────────────────────────────
    if prediction == "ABNORMAL":
        st.error("🔴 **ABNORMAL** — Possible cardiac abnormality detected")
    elif prediction == "BORDERLINE":
        st.warning(
            "🟡 **BORDERLINE** — Elevated probability of abnormality detected. "
            "Further clinical evaluation recommended."
        )
    else:
        st.success("🟢 **NORMAL** — No abnormality detected")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean probability",  f"{mean_prob:.3f}")
    c2.metric("Max probability",   f"{max_prob:.3f}")
    c3.metric("Decision threshold", f"{THRESHOLD:.3f}")
    c4.metric("Abnormal windows",
            f"{sum(1 for p in all_probs if p >= THRESHOLD)} / {len(all_probs)}")

    # per-window breakdown
    with st.expander("Per-window probabilities"):
        for i, p in enumerate(all_probs):
            label = "abnormal" if p >= THRESHOLD else "normal"
            st.write(f"Window {i+1}: p={p:.3f} → **{label}**")

    st.divider()

    # ── Grad-CAM for most representative window ───────────────
    st.subheader("Grad-CAM Explainability")
    st.markdown(
        "The heatmap shows which time-frequency regions of the spectrogram "
        "most influenced the model's prediction. "
        "**Red/yellow = high influence, blue = low influence.**"
    )

    # pick the window whose prob is closest to the mean (most representative)
    rep_idx = int(np.argmin(np.abs(np.array(all_probs) - mean_prob)))
    fig = make_overlay_figure(
        all_logmels[rep_idx], all_cams[rep_idx],
        f"Window {rep_idx+1} of {len(windows)} "
        f"(p={all_probs[rep_idx]:.3f}, most representative)"
    )
    st.pyplot(fig)
    plt.close(fig)

    # most abnormal window (highest prob) if abnormal predicted
    if prediction == "ABNORMAL":
        st.markdown("**Most abnormal window:**")
        max_idx = int(np.argmax(all_probs))
        fig2 = make_overlay_figure(
            all_logmels[max_idx], all_cams[max_idx],
            f"Window {max_idx+1} — highest p={all_probs[max_idx]:.3f}"
        )
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()
    st.caption(
        "Model: CNN-BiLSTM with attention | "
        "Trained on PhysioNet/CinC 2016 + CirCor DigiScope 2022 | "
        "Test AUC: 0.9567 | "
        "For research use only"
    )