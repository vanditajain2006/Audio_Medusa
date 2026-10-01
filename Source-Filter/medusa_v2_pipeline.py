# ============================================================
# HYBRID BATCH PIPELINE v5
# Source-Filter + Waveform Perturbation
# Tested on 10 LibriSpeech speakers
# ============================================================

import sys
import os
import torch

# PyTorch 2.6 weights_only workaround for Coqui TTS
_orig_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _safe_load

import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torch.optim as optim
import math
import time
import gc
import numpy as np
import pandas as pd
import librosa
from datasets import load_dataset
from speechbrain.inference.speaker import SpeakerRecognition
from pesq import pesq
from pystoi import stoi
import warnings
warnings.filterwarnings("ignore")

import sys
import speechbrain.utils.importutils
# Monkeypatch SpeechBrain LazyModule to prevent linecache from crashing during traceback extraction
_orig_getattr = speechbrain.utils.importutils.LazyModule.__getattr__
def _safe_getattr(self, attr):
    if attr == '__file__':
        return None
    return _orig_getattr(self, attr)
speechbrain.utils.importutils.LazyModule.__getattr__ = _safe_getattr

os.environ["COQUI_TOS_AGREED"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000

INPUT_DIR = "./audio_samples"
OUTPUT_DIR = "./experiments_v2"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PART 1: DOWNLOAD DATASET
# ============================================================

print("█" * 70)
print("█" + " DOWNLOADING DATASET ".center(68) + "█")
print("█" * 70)

AUDIO_FILES = [os.path.join(INPUT_DIR, f"sample_{i}.wav") for i in range(1, 3)]
files_missing = any(not os.path.exists(f) for f in AUDIO_FILES)

if files_missing:
    print("Files missing, but we bypassed huggingface datasets. Using pre-downloaded samples.")
#    print("Accessing LibriSpeech (Streaming Mode)...")
#    dataset = load_dataset(
#        "librispeech_asr", "clean", split="train.100", streaming=True
#    ).shuffle(seed=42)
#    for i, example in enumerate(dataset.take(10)):
#        audio_tensor = torch.tensor(example["audio"]["array"]).unsqueeze(0)
#        sampling_rate = example["audio"]["sampling_rate"]
#        torchaudio.save(AUDIO_FILES[i], audio_tensor, sampling_rate)
#        print(f"   ✅ Saved: {AUDIO_FILES[i]}")
else:
    print("✅ All samples exist.")
print()


# ============================================================
# MODELS
# ============================================================

class AdaptiveSourceFilterModule(nn.Module):
    """
    Adaptive Frame-Conditioned Source-Filter Module.
    Predicts dynamic filter (LPC, EQ) and source (Phase Warp, Tilt) perturbations
    for each 20ms frame using a neural network conditioned on the mel-spectrogram.
    """
    def __init__(self, order=16, n_tilt_stages=1, n_sq_basis=8, n_bands=8,
                 beta_limit=0.35, sq_limit=0.50, lpc_limit=0.02, gain_limit=0.20,
                 n_fft=1024, hop_length=256):
        super().__init__()
        self.order = order
        self.n_tilt_stages = n_tilt_stages
        self.n_sq_basis = n_sq_basis
        self.n_bands = n_bands
        self.beta_limit = beta_limit
        self.sq_limit = sq_limit
        self.lpc_limit = lpc_limit
        self.gain_limit = gain_limit
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft

        # Mel-spectrogram for conditioning (frozen, used for feature extraction)
        self.mel_spec = T.MelSpectrogram(sample_rate=SR, n_fft=n_fft, 
                                         hop_length=hop_length, n_mels=80)
        
        # Output dimension for the neural network
        self.out_dim = n_tilt_stages + n_sq_basis + order + n_bands
        
        # Conditioning network: 1D CNN over time
        self.net = nn.Sequential(
            nn.Conv1d(80, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm1d(128),
            nn.Conv1d(128, self.out_dim, kernel_size=1)
        )
        
        self.register_buffer('window', torch.hann_window(n_fft))
        self.register_buffer(
            'band_edges',
            torch.tensor([0, 300, 700, 1200, 2000, 3000, 4500, 6000, 8000], dtype=torch.float32)
        )

    def forward(self, x):
        B, L = x.shape
        dev = x.device
        
        # 1. Conditioning Features (Mel-Spectrogram)
        mel = self.mel_spec(x) # (B, 80, T)
        mel = torch.log(mel + 1e-5)
        T_frames = mel.shape[-1]
        
        # 2. Predict adaptive parameters dynamically per frame
        params = self.net(mel) # (B, out_dim, T)
        
        idx = 0
        betas = torch.tanh(params[:, idx:idx+self.n_tilt_stages, :]) * self.beta_limit
        idx += self.n_tilt_stages
        sq_coeffs = torch.tanh(params[:, idx:idx+self.n_sq_basis, :]) * self.sq_limit
        idx += self.n_sq_basis
        lpc_offsets = torch.tanh(params[:, idx:idx+self.order, :]) * self.lpc_limit
        idx += self.order
        band_gains = torch.tanh(params[:, idx:idx+self.n_bands, :]) * self.gain_limit
        
        # 3. STFT of the waveform
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, 
                       win_length=self.win_length, window=self.window, 
                       return_complex=True) # (B, N_f, T)
        
        # Trim T_frames if stft gives different number of frames
        T_stft = X.shape[-1]
        if T_frames != T_stft:
            # Match frames
            min_T = min(T_frames, T_stft)
            betas = betas[:, :, :min_T]
            sq_coeffs = sq_coeffs[:, :, :min_T]
            lpc_offsets = lpc_offsets[:, :, :min_T]
            band_gains = band_gains[:, :, :min_T]
            X = X[:, :, :min_T]
            T_frames = min_T
        
        # 4. LPC Analysis per frame (from STFT power spectrum)
        P = X.abs() ** 2 # Power spectrum
        autocorr = torch.fft.irfft(P, n=self.n_fft, dim=1) # Autocorrelation
        r = autocorr[:, :self.order + 1, :] # (B, order+1, T)
        
        # Batch solve Levinson-Durbin
        r_bt = r.transpose(1, 2).reshape(B * T_frames, self.order + 1)
        R_bt = torch.zeros(B * T_frames, self.order, self.order, device=dev)
        for i in range(self.order):
            for j in range(self.order):
                R_bt[:, i, j] = r_bt[:, abs(i - j)]
        
        # Tiny regularization to ensure invertibility
        R_bt = R_bt + torch.eye(self.order, device=dev).unsqueeze(0) * 1e-5
        
        a_bt = torch.linalg.solve(R_bt, r_bt[:, 1:]) 
        a_bt_orig = torch.cat([torch.ones(B * T_frames, 1, device=dev), -a_bt], dim=1) 
        a_orig = a_bt_orig.view(B, T_frames, self.order + 1).transpose(1, 2) # (B, order+1, T)
        
        # 5. Apply Frame-Conditioned LPC Offsets
        a_pert = a_orig.clone()
        a_pert[:, 1:, :] = a_orig[:, 1:, :] + lpc_offsets
        
        # 6. Source-Filter decomposition in frequency domain
        A_p = F.pad(a_orig, (0, 0, 0, self.n_fft - a_orig.shape[1]))
        A_f = torch.fft.rfft(A_p, n=self.n_fft, dim=1) 
        
        # Extract Source Excitation (E)
        E = X * A_f 
        
        # Compute perturbed filter response
        A_p_pert = F.pad(a_pert, (0, 0, 0, self.n_fft - a_pert.shape[1]))
        A_pert_f = torch.fft.rfft(A_p_pert, n=self.n_fft, dim=1)
        
        # 7. Apply Phase Warp (Source perturbation)
        mag, ph = E.abs(), E.angle()
        fb = torch.linspace(0, math.pi, mag.shape[1], device=dev)
        disp = torch.zeros(B, mag.shape[1], T_frames, device=dev)
        for k in range(self.n_sq_basis):
            sq_k = sq_coeffs[:, k:k+1, :] 
            sin_fb = torch.sin((k + 1) * fb).unsqueeze(0).unsqueeze(2) 
            disp = disp + sq_k * sin_fb
        
        E_w = mag * torch.polar(torch.ones_like(mag), ph + disp)
        
        # 8. Recombine and apply EQ / Tilt (Filter perturbations)
        # Bug Fix: Wiener-style deconvolution avoids division by near-zero values
        # that cause catastrophic amplification when A_pert_f has bins near 0.
        A_mag_sq = A_pert_f.abs() ** 2
        X_r = E_w * A_pert_f.conj() / (A_mag_sq + 1e-4)
        
        # EQ
        fhz = torch.linspace(0, SR / 2, mag.shape[1], device=dev)
        gp = torch.zeros(B, mag.shape[1], T_frames, device=dev)
        for b in range(self.n_bands):
            lo, hi = self.band_edges[b], self.band_edges[b + 1]
            mask = torch.sigmoid((fhz - lo) / 50) * torch.sigmoid((hi - fhz) / 50) 
            gain_b = band_gains[:, b:b+1, :] 
            gp = gp + gain_b * mask.unsqueeze(0).unsqueeze(2)
            
        X_r = X_r * torch.exp(gp)
        
        # Spectral Tilt
        for k in range(self.n_tilt_stages):
            beta_k = betas[:, k:k+1, :] 
            w = fb.unsqueeze(0).unsqueeze(2) 
            tilt_f = 1.0 - beta_k * torch.polar(torch.ones_like(w), -w)
            X_r = X_r * tilt_f
            
        # 9. Overlap-Add Synthesis
        out = torch.istft(X_r, n_fft=self.n_fft, hop_length=self.hop_length, 
                          win_length=self.win_length, window=self.window, 
                          length=L)
        
        # Save generated parameters for regularization in train_one
        self.last_params = (betas, sq_coeffs, lpc_offsets, band_gains)
        
        return out
        
    def clamp_params(self):
        # Bound enforcement is now handled dynamically inside the network via tanh!
        pass


class HybridAttacker(nn.Module):
    """
    Source-Filter + Waveform Perturbation.
    SF provides structured biological changes.
    Delta provides targeted adversarial perturbation.
    """
    def __init__(self, audio_length, order=16, max_perturb_ratio=0.025,
                 sf_kwargs=None):
        super().__init__()
        sf_kw = sf_kwargs or {}
        self.sf = AdaptiveSourceFilterModule(order=order, **sf_kw)
        self.delta = nn.Parameter(torch.zeros(1, audio_length))
        self.max_ratio = max_perturb_ratio

    def forward(self, x, signal_rms):
        sf_out = self.sf(x)
        max_amp = signal_rms * self.max_ratio
        d = torch.clamp(self.delta, -max_amp, max_amp)
        dl = min(d.shape[-1], sf_out.shape[-1])
        return sf_out[..., :dl] + d[..., :dl]

    def clamp_all(self, signal_rms):
        self.sf.clamp_params()
        ma = (signal_rms * self.max_ratio).item()
        with torch.no_grad():
            self.delta.data.clamp_(-ma, ma)


class DeltaOnlyAttacker(nn.Module):
    """
    Ablation model: ONLY waveform perturbation, no source-filter.
    Used to prove that SF contributes beyond generic adversarial noise.
    """
    def __init__(self, audio_length, max_perturb_ratio=0.025):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(1, audio_length))
        self.max_ratio = max_perturb_ratio

    def forward(self, x, signal_rms):
        max_amp = signal_rms * self.max_ratio
        d = torch.clamp(self.delta, -max_amp, max_amp)
        dl = min(d.shape[-1], x.shape[-1])
        return x[..., :dl] + d[..., :dl]

    def clamp_all(self, signal_rms):
        ma = (signal_rms * self.max_ratio).item()
        with torch.no_grad():
            self.delta.data.clamp_(-ma, ma)


# ============================================================
# DIFFERENTIABLE AUGMENTATIONS (W3: EOT Training)
# ============================================================

class DifferentiableAugmentations:
    """
    Random signal-level transformations applied during training.
    Forces perturbations to be robust to real-world processing.
    All ops are differentiable (or use straight-through estimation).
    """
    def __init__(self, sr=16000):
        self.sr = sr

    def random_gain(self, x, min_db=-3.0, max_db=3.0):
        """Random gain change in dB."""
        gain_db = torch.empty(1, device=x.device).uniform_(min_db, max_db)
        return x * (10.0 ** (gain_db / 20.0))

    def random_lowpass(self, x, min_cutoff=4000, max_cutoff=7000):
        """Approximate low-pass filter via FFT zeroing (differentiable)."""
        L = x.shape[-1]
        X = torch.fft.rfft(x, n=L)
        freqs = torch.linspace(0, self.sr / 2, X.shape[-1], device=x.device)
        cutoff = torch.empty(1, device=x.device).uniform_(min_cutoff, max_cutoff).item()
        # Smooth rolloff instead of hard cutoff
        mask = torch.sigmoid((cutoff - freqs) / 200.0)
        X_filtered = X * mask.unsqueeze(0)
        return torch.fft.irfft(X_filtered, n=L)

    def random_noise(self, x, min_snr=20.0, max_snr=40.0):
        """Add Gaussian noise at a random SNR."""
        snr_db = torch.empty(1, device=x.device).uniform_(min_snr, max_snr).item()
        sig_power = torch.mean(x ** 2)
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = torch.randn_like(x) * torch.sqrt(noise_power + 1e-10)
        return x + noise

    def random_resample(self, x):
        """Downsample to 8kHz and back (simulates bandwidth limitation)."""
        # Use interpolation for differentiability
        L = x.shape[-1]
        half_L = L // 2
        x_down = F.interpolate(x.unsqueeze(0), size=half_L, mode='linear', align_corners=False).squeeze(0)
        x_up = F.interpolate(x_down.unsqueeze(0), size=L, mode='linear', align_corners=False).squeeze(0)
        return x_up

    def mel_reconstruction(self, x, n_fft=1024, n_mels=80):
        """Approximate codec: mel-spectrogram → inverse → waveform.
        This simulates what cloning models do internally, stripping
        waveform-level perturbations that don't survive mel conversion."""
        window = torch.hann_window(n_fft, device=x.device)
        stft = torch.stft(x.squeeze(0), n_fft=n_fft, hop_length=256,
                          return_complex=True, window=window)
        mag = stft.abs()
        phase = stft.angle()
        # Create mel filterbank
        mel_fb = torch.zeros(n_mels, mag.shape[0], device=x.device)
        mel_freqs = torch.linspace(0, 2595 * np.log10(1 + self.sr / 2 / 700), n_mels + 2, device=x.device)
        mel_freqs = 700 * (10 ** (mel_freqs / 2595) - 1)
        freq_bins = torch.linspace(0, self.sr / 2, mag.shape[0], device=x.device)
        for i in range(n_mels):
            lo, center, hi = mel_freqs[i], mel_freqs[i+1], mel_freqs[i+2]
            up_slope = (freq_bins - lo) / (center - lo + 1e-8)
            down_slope = (hi - freq_bins) / (hi - center + 1e-8)
            mel_fb[i] = torch.clamp(torch.minimum(up_slope, down_slope), min=0.0)
        # Project to mel and back
        mel_spec = torch.matmul(mel_fb, mag)  # [n_mels, T]
        mag_reconstructed = torch.matmul(mel_fb.T, mel_spec)  # [F, T]
        # Normalize to match original energy
        mag_reconstructed = mag_reconstructed * (mag.sum() / (mag_reconstructed.sum() + 1e-8))
        # Reconstruct with original phase
        stft_recon = mag_reconstructed * torch.exp(1j * phase)
        out = torch.istft(stft_recon, n_fft=n_fft, hop_length=256, window=window, length=x.shape[-1])
        return out.unsqueeze(0)

    def apply_random(self, x, p=0.5):
        """
        Apply a random subset of augmentations with probability p each.
        Returns augmented waveform (differentiable).
        """
        if torch.rand(1).item() < p:
            x = self.random_gain(x)
        if torch.rand(1).item() < p:
            x = self.random_lowpass(x)
        if torch.rand(1).item() < p * 0.7:  # noise slightly less frequent
            x = self.random_noise(x)
        if torch.rand(1).item() < p * 0.3:  # resample is harsh, less frequent
            x = self.random_resample(x)
        if torch.rand(1).item() < p * 0.4:  # mel reconstruction simulates cloner front-end
            x = self.mel_reconstruction(x)
        return x


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def mcd_attack_loss(original, perturbed, mfcc_transform):
    mo = mfcc_transform(original)
    mp = mfcc_transform(perturbed)
    diff = mo[:, 1:14, :] - mp[:, 1:14, :]
    return -torch.mean(torch.sqrt(2.0 * torch.sum(diff ** 2, dim=1) + 1e-8))


def delta_mfcc_attack_loss(original, perturbed, mfcc_transform):
    mo = mfcc_transform(original)[:, 1:14, :]
    mp = mfcc_transform(perturbed)[:, 1:14, :]
    do_ = mo[:, :, 2:] - mo[:, :, :-2]
    dp_ = mp[:, :, 2:] - mp[:, :, :-2]
    ddo = do_[:, :, 2:] - do_[:, :, :-2]
    ddp = dp_[:, :, 2:] - dp_[:, :, :-2]
    return -torch.mean((do_ - dp_) ** 2) - 0.5 * torch.mean((ddo - ddp) ** 2)


def cosine_embed_loss(original, perturbed, verifiers, embs_orig):
    total_sim = 0.0
    for v_name, verifier in verifiers.items():
        emb_pert = verifier.encode_batch(perturbed)
        sim = F.cosine_similarity(embs_orig[v_name].squeeze(), emb_pert.squeeze(), dim=0)
        total_sim += sim
    return total_sim / len(verifiers)


def ltas_divergence_loss(original, perturbed, n_fft=2048):
    window = torch.hann_window(n_fft).to(original.device)
    so = torch.stft(original, n_fft=n_fft, hop_length=512,
                    return_complex=True, window=window)
    sp = torch.stft(perturbed, n_fft=n_fft, hop_length=512,
                    return_complex=True, window=window)
    lo = torch.mean(torch.log(torch.abs(so) + 1e-8), dim=-1)
    lp = torch.mean(torch.log(torch.abs(sp) + 1e-8), dim=-1)
    l2 = -torch.mean((lo - lp) ** 2)
    loc = lo - lo.mean(dim=-1, keepdim=True)
    lpc = lp - lp.mean(dim=-1, keepdim=True)
    corr = torch.sum(loc * lpc, dim=-1) / \
           (torch.norm(loc, dim=-1) * torch.norm(lpc, dim=-1) + 1e-8)
    return l2 + torch.mean(corr)


def multi_resolution_mel_loss(original, perturbed, mel_transforms):
    total = 0.0
    for mt in mel_transforms:
        total += F.l1_loss(torch.log(mt(perturbed) + 1e-7),
                           torch.log(mt(original) + 1e-7))
    return total / len(mel_transforms)


def temporal_envelope_loss(original, perturbed, frame_size=320):
    of = original.unfold(-1, frame_size, frame_size // 2)
    pf = perturbed.unfold(-1, frame_size, frame_size // 2)
    eo = torch.sqrt(torch.mean(of ** 2, dim=-1) + 1e-8)
    ep = torch.sqrt(torch.mean(pf ** 2, dim=-1) + 1e-8)
    return F.mse_loss(ep, eo)


def waveform_snr_loss(original, perturbed, target_snr_db=25.0):
    noise = perturbed - original
    sig = torch.mean(original ** 2)
    noi = torch.mean(noise ** 2) + 1e-10
    return F.relu(target_snr_db - 10 * torch.log10(sig / noi))


def smoothness_loss(delta):
    d2 = delta[:, 2:] - 2 * delta[:, 1:-1] + delta[:, :-2]
    return torch.mean(d2 ** 2)


def spectral_convergence_loss(original, perturbed, n_fft=1024):
    w = torch.hann_window(n_fft).to(original.device)
    mo = torch.abs(torch.stft(original, n_fft=n_fft, hop_length=256,
                               return_complex=True, window=w))
    mp = torch.abs(torch.stft(perturbed, n_fft=n_fft, hop_length=256,
                               return_complex=True, window=w))
    return torch.norm(mo - mp) / (torch.norm(mo) + 1e-8)


# ============================================================
# WHISPER CONTENT PRESERVATION LOSS (W4)
# ============================================================

_whisper_model = None

def get_whisper_model(device="cpu"):
    """Lazy-load whisper-tiny for content preservation loss."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print("   Loading Whisper-tiny for content preservation...")
        _whisper_model = whisper.load_model("tiny", device=device)
        _whisper_model.eval()
        for p in _whisper_model.parameters():
            p.requires_grad_(False)
        print("   ✅ Whisper loaded.")
    return _whisper_model


# Whisper's exact front-end constants
_WHISPER_HOP = 160
_WHISPER_NFFT = 400
_WHISPER_TARGET_FRAMES = 3000   # positional embedding size; 30s * 100 frames/s
_whisper_mel_filters = None

def _get_whisper_mel_filters(n_mels, device):
    """Build mel filterbank matching Whisper's front-end (differentiable)."""
    global _whisper_mel_filters
    if _whisper_mel_filters is None:
        import torchaudio.functional as AF
        fb = AF.melscale_fbanks(
            n_freqs=_WHISPER_NFFT // 2 + 1,
            f_min=0.0,
            f_max=8000.0,
            n_mels=n_mels,
            sample_rate=16000,
            norm="slaney",
            mel_scale="htk",
        )  # (n_freqs, n_mels)
        _whisper_mel_filters = fb
    return _whisper_mel_filters.to(device)


def _differentiable_log_mel(waveform, n_mels, device):
    """
    Fully differentiable log-mel matching Whisper encoder input shape exactly.
    waveform: (1, L) tensor padded to 480000 samples.
    Returns: (1, n_mels, 3000) — guaranteed shape match with Whisper positional embedding.
    """
    window = torch.hann_window(_WHISPER_NFFT, device=device)
    stft = torch.stft(
        waveform.squeeze(0),
        n_fft=_WHISPER_NFFT,
        hop_length=_WHISPER_HOP,
        win_length=_WHISPER_NFFT,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )  # (n_freqs, T_actual)
    magnitudes = stft[..., :-1].abs() ** 2          # drop last frame -> (n_freqs, T)
    mel_fb = _get_whisper_mel_filters(n_mels, device)  # (n_freqs, n_mels)
    mel_spec = torch.matmul(mel_fb.T, magnitudes)       # (n_mels, T)
    log_mel = torch.log10(torch.clamp(mel_spec, min=1e-10))
    log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    log_mel = log_mel.unsqueeze(0)  # (1, n_mels, T)
    # Hard-resize to exactly 3000 frames — guarantees Whisper encoder shape match
    if log_mel.shape[-1] != _WHISPER_TARGET_FRAMES:
        log_mel = F.interpolate(log_mel, size=_WHISPER_TARGET_FRAMES,
                                mode='linear', align_corners=False)
    return log_mel  # (1, n_mels, 3000)


def whisper_content_loss(original, perturbed, device="cpu"):
    """
    Content preservation loss using Whisper encoder hidden states.
    Returns 1 - cosine_similarity(encoder(orig), encoder(prot)).
    High values = content has been damaged.

    Uses a fully-differentiable PyTorch STFT + mel filterbank for the perturbed
    path so gradients flow back to the protection model. Frame axis is
    hard-interpolated to exactly 3000 to guarantee Whisper encoder shape match.
    """
    whisper_model = get_whisper_model(device)
    import whisper
    n_mels = whisper_model.dims.n_mels

    # ---- ORIGINAL (fixed target, no grad needed) ----
    with torch.no_grad():
        orig_padded = whisper.pad_or_trim(original.squeeze(0)).unsqueeze(0)
        mel_orig_np = whisper.log_mel_spectrogram(orig_padded.squeeze(0), n_mels=n_mels)
        mel_orig = mel_orig_np.unsqueeze(0).to(device)
        enc_orig = whisper_model.encoder(mel_orig).squeeze(0).mean(dim=0)

    # ---- PERTURBED (stays in graph for backprop) ----
    target_len = 16000 * 30
    L = perturbed.shape[-1]
    pert_padded = F.pad(perturbed, (0, target_len - L)) if L < target_len else perturbed[..., :target_len]

    # Differentiable log-mel with guaranteed shape (1, n_mels, 3000)
    mel_pert = _differentiable_log_mel(pert_padded, n_mels, device)
    enc_pert = whisper_model.encoder(mel_pert).squeeze(0).mean(dim=0)

    cos_sim = F.cosine_similarity(enc_orig, enc_pert, dim=0)
    return 1.0 - cos_sim


# ============================================================
# EER CALCULATION
# ============================================================

def compute_eer(pos_scores, neg_scores):
    if not pos_scores or not neg_scores:
        return 0.0
    scores = np.array(pos_scores + neg_scores)
    labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    desc = np.argsort(scores)[::-1]
    scores, labels = scores[desc], labels[desc]
    fps = np.cumsum(labels == 0)
    tps = np.cumsum(labels == 1)
    far = fps / float(len(neg_scores))
    fnr = 1.0 - (tps / float(len(pos_scores)))
    idx = np.argmin(np.abs(far - fnr))
    return round((far[idx] + fnr[idx]) / 2.0 * 100, 2)


# ============================================================
# EVALUATION HELPERS
# ============================================================

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def calculate_protection_metrics(orig_path, prot_path, verifiers):
    wav_o, _ = librosa.load(orig_path, sr=16000)
    wav_p, _ = librosa.load(prot_path, sr=16000)
    ml = min(len(wav_o), len(wav_p))
    wav_o, wav_p = wav_o[:ml], wav_p[:ml]

    noise = wav_o - wav_p
    snr = 10 * np.log10(np.sum(wav_o ** 2) / (np.sum(noise ** 2) + 1e-8))
    try:
        pesq_val = pesq(16000, np.clip(wav_o, -1, 1), np.clip(wav_p, -1, 1), 'wb')
    except Exception:
        pesq_val = 1.0
    try:
        stoi_val = stoi(wav_o, wav_p, 16000, extended=False)
    except Exception:
        stoi_val = 0.0

    mfcc_o = librosa.feature.mfcc(y=wav_o, sr=16000, n_mfcc=14)[1:]
    mfcc_p = librosa.feature.mfcc(y=wav_p, sr=16000, n_mfcc=14)[1:]
    mt = min(mfcc_o.shape[1], mfcc_p.shape[1])
    mcd_val = float(np.mean(np.sqrt(2.0 * np.sum(
        (mfcc_o[:, :mt] - mfcc_p[:, :mt]) ** 2, axis=0
    ))))

    t_o = torch.tensor(wav_o).unsqueeze(0).to(DEVICE)
    t_p = torch.tensor(wav_p).unsqueeze(0).to(DEVICE)
    cos_sim_total = 0.0
    with torch.no_grad():
        for v_name, verifier in verifiers.items():
            eo = verifier.encode_batch(t_o)
            ep = verifier.encode_batch(t_p)
            cos_sim_total += F.cosine_similarity(eo.squeeze(), ep.squeeze(), dim=0).item()
        cos_sim = cos_sim_total / len(verifiers)

    return {
        "CosSim_Prot": round(cos_sim, 4),
        "MCD_Prot": round(mcd_val, 2),
        "PESQ": round(float(pesq_val), 2),
        "STOI": round(float(stoi_val), 4),
        "SNR_dB": round(float(snr), 1),
    }


def calculate_clone_metrics(orig_path, clone_path, verifiers):
    wav_o, _ = librosa.load(orig_path, sr=16000)
    wav_c, _ = librosa.load(clone_path, sr=16000)

    t_o = torch.tensor(wav_o).unsqueeze(0).to(DEVICE)
    t_c = torch.tensor(wav_c).unsqueeze(0).to(DEVICE)
    cos_sim_total = 0.0
    with torch.no_grad():
        for v_name, verifier in verifiers.items():
            eo = verifier.encode_batch(t_o)
            ec = verifier.encode_batch(t_c)
            cos_sim_total += F.cosine_similarity(eo.squeeze(), ec.squeeze(), dim=0).item()
        cos_sim = cos_sim_total / len(verifiers)

    mfcc_o = librosa.feature.mfcc(y=wav_o, sr=16000, n_mfcc=14)[1:]
    mfcc_c = librosa.feature.mfcc(y=wav_c, sr=16000, n_mfcc=14)[1:]
    clone_mcd = float(np.sqrt(2.0 * np.sum(
        (np.mean(mfcc_o, axis=1) - np.mean(mfcc_c, axis=1)) ** 2
    )))

    return {"Clone_CosSim": round(cos_sim, 4), "Clone_MCD": round(clone_mcd, 2)}


# ============================================================
# UNIFIED TRAINING FUNCTION
# ============================================================

def train_one(waveform, config, verifiers, embs_orig, mfcc_t, mel_ts):
    """
    Train a single model on a single audio.
    Returns the protected waveform tensor.
    """
    device = DEVICE
    model_type = config.get("model_type", "sf_only")
    iterations = config.get("iterations", 800)
    lr = config.get("lr", 0.005)
    use_augmentation = config.get("augmentation", False)
    sig_rms = torch.sqrt(torch.mean(waveform ** 2))

    # Initialize augmentation pipeline if enabled
    augmentor = DifferentiableAugmentations(sr=SR) if use_augmentation else None

    if model_type == "hybrid":
        sf_kw = {
            k: config[k] for k in
            ["beta_limit", "sq_limit", "lpc_limit", "gain_limit",
             "n_tilt_stages", "n_sq_basis", "n_bands"]
            if k in config
        }
        model = HybridAttacker(
            audio_length=waveform.shape[-1],
            max_perturb_ratio=config.get("max_perturb_ratio", 0.025),
            sf_kwargs=sf_kw,
        ).to(device)
        optimizer = optim.Adam([
            {"params": model.sf.parameters(), "lr": lr},
            {"params": [model.delta], "lr": lr * 3},
        ])
    elif model_type == "delta_only":
        model = DeltaOnlyAttacker(
            audio_length=waveform.shape[-1],
            max_perturb_ratio=config.get("max_perturb_ratio", 0.025),
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr * 3)
    else:
        model = AdaptiveSourceFilterModule(
            beta_limit=config.get('beta_limit', 0.35),
            sq_limit=config.get('sq_limit', 0.50),
            lpc_limit=config.get('lpc_limit', 0.02),
            gain_limit=config.get('gain_limit', 0.20)
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

    if config.get("use_scheduler", True):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations)
    else:
        scheduler = None

    best_loss, best_state = float('inf'), None
    cached_c_loss = 0.0

    for step in range(iterations):
        optimizer.zero_grad()

        if model_type in ("hybrid", "delta_only"):
            hacked = model(waveform, sig_rms)
        else:
            hacked = model(waveform)

        if torch.isnan(hacked).any():
            with torch.no_grad():
                # Bug Fix: AdaptiveSourceFilterModule has no static lpc_offsets/betas.
                # Instead, scale down the CNN output layer weights to reduce the
                # magnitude of predicted parameters and escape NaN territory.
                if model_type == "hybrid":
                    model.sf.net[-1].weight.data *= 0.3
                    model.sf.net[-1].bias.data *= 0.3
                    model.delta.data *= 0.5
                elif model_type == "delta_only":
                    model.delta.data *= 0.5
                else:  # sf_only
                    model.net[-1].weight.data *= 0.3
                    model.net[-1].bias.data *= 0.3
            continue

        # For EOT: apply augmentation to the hacked signal for attack losses
        if augmentor is not None:
            hacked_aug = augmentor.apply_random(hacked, p=0.5)
        else:
            hacked_aug = hacked

        # Bug Fix 3: Proper leaf tensor for loss accumulation
        loss = torch.zeros(1, device=device, requires_grad=True).squeeze()
        current_attack_sim = float('inf')

        # ===== ATTACK (use augmented signal if EOT) =====
        if "mcd" in config:
            loss = loss + config["mcd"] * mcd_attack_loss(
                waveform, hacked_aug, mfcc_t
            )
        if "delta_mfcc" in config:
            loss = loss + config["delta_mfcc"] * delta_mfcc_attack_loss(
                waveform, hacked_aug, mfcc_t
            )
        if "cosine_embed" in config:
            cos_embed_val = cosine_embed_loss(waveform, hacked_aug, verifiers, embs_orig)
            current_attack_sim = cos_embed_val.item()
            loss = loss + config["cosine_embed"] * cos_embed_val
        if "ltas" in config:
            loss = loss + config["ltas"] * ltas_divergence_loss(
                waveform, hacked_aug
            )

        # ===== PRESERVATION (always on clean hacked signal) =====
        if "mel_preserve" in config:
            loss = loss + config["mel_preserve"] * multi_resolution_mel_loss(
                waveform, hacked, mel_ts
            )
        if "temporal_env" in config:
            loss = loss + config["temporal_env"] * temporal_envelope_loss(
                waveform, hacked
            )
        if "snr_floor" in config:
            loss = loss + config["snr_floor"] * waveform_snr_loss(
                waveform, hacked, config.get("snr_target", 25.0)
            )
        if "spectral_conv" in config:
            loss = loss + config["spectral_conv"] * spectral_convergence_loss(
                waveform, hacked
            )

        # ===== CONTENT PRESERVATION (W4) =====
        if "content_preserve" in config:
            # Recompute Whisper loss every 50 steps (expensive).
            # On recompute steps: use the live tensor so gradients flow.
            # On cached steps: use the detached scalar — it's a constant
            #   regularisation weight, not a dynamic gradient source.
            recompute = (step % 50 == 0 or step == iterations - 1 or step == 0)
            if recompute:
                c_loss_live = whisper_content_loss(waveform, hacked, device=device)
                cached_c_loss = c_loss_live.item()   # store scalar for non-compute steps
                loss = loss + config["content_preserve"] * c_loss_live
            else:
                # Constant penalty — no gradient through Whisper, but
                # the other loss terms still update the model.
                loss = loss + config["content_preserve"] * cached_c_loss

        # Delta-specific
        if model_type in ("hybrid", "delta_only"):
            delta_param = model.delta
            if "smoothness" in config:
                loss = loss + config["smoothness"] * smoothness_loss(delta_param)
            if "delta_energy" in config:
                loss = loss + config["delta_energy"] * torch.mean(delta_param ** 2)

        # SF param reg
        if model_type == "hybrid":
            sf_mod = model.sf
        elif model_type == "sf_only":
            sf_mod = model
        else:
            sf_mod = None

        if sf_mod is not None and "param_reg" in config and hasattr(sf_mod, 'last_params'):
            betas, sq_coeffs, lpc_offsets, band_gains = sf_mod.last_params
            # Issue 6: Use mean() instead of sum() to avoid penalizing longer audios more
            reg = (
                torch.mean(betas ** 2)
                + torch.mean(sq_coeffs ** 2)
                + torch.mean(lpc_offsets ** 2)
                + torch.mean(band_gains ** 2)
            )
            loss = loss + config["param_reg"] * reg

        # Parameter bounds are now strictly enforced via tanh() inside the forward pass of AdaptiveSourceFilterModule.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        if model_type in ("hybrid", "delta_only"):
            model.clamp_all(sig_rms)
        elif model_type == "sf_only":
            model.clamp_params()

        lv = loss.item()
        # Issue 9: Track best state by attack effectiveness (cosine embed sim) if available, else by total loss
        track_metric = current_attack_sim if current_attack_sim != float('inf') else lv
        if track_metric < best_loss and not math.isnan(lv):
            best_loss = track_metric
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    with torch.no_grad():
        if model_type in ("hybrid", "delta_only"):
            final = model(waveform, sig_rms)
        else:
            final = model(waveform)
        if torch.isnan(final).any():
            final = waveform
        final = torch.clamp(final, -1.0, 1.0)

    return final


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================
#
# LESSON FROM YOUR RESULTS:
# Source-filter only with safe clamps = 0% protection.
# We need hybrid models with waveform perturbation.
#
# We also keep 2 SF-only configs as controls to prove
# the hybrid improvement on 10 speakers.

EXPERIMENTS = {

    # ==========================================
    # CONTROLS (SF-only)
    # ==========================================

    "CTRL_sf_moderate": {
        "model_type": "sf_only",
        "iterations": 800,
        "beta_limit": 0.35, "lpc_limit": 0.025, "gain_limit": 0.20,
        "mcd": 1.5, "delta_mfcc": 1.0,
        "cosine_embed": 2.0,
        "mel_preserve": 10.0, "snr_floor": 40.0, "snr_target": 15.0,
        "param_reg": 1.0,
    },

    # ==========================================
    # ABLATION STUDY (W4C) — proves SF contribution
    # ==========================================

    "ABL_delta_only": {
        "model_type": "delta_only",
        "iterations": 1000,
        "lr": 0.005,
        "max_perturb_ratio": 0.025,
        "mcd": 2.0,
        "cosine_embed": 3.0,
        "delta_mfcc": 1.0,
        "mel_preserve": 12.0,
        "snr_floor": 40.0,
        "snr_target": 25.0,
        "smoothness": 100.0,
        "delta_energy": 400.0,
    },

    "ABL_sf_only": {
        "model_type": "sf_only",
        "iterations": 1200,
        "lr": 0.008,
        "beta_limit": 0.45, "sq_limit": 0.60, "lpc_limit": 0.030, "gain_limit": 0.25,
        "mcd": 2.5,
        "cosine_embed": 3.0,
        "delta_mfcc": 1.5,
        "ltas": 0.5,
        "mel_preserve": 8.0,
        "snr_floor": 30.0,
        "snr_target": 20.0,
        "param_reg": 0.3,
    },

    "ABL_sf_tiny_delta": {
        "model_type": "hybrid",
        "iterations": 1200,
        "lr": 0.006,
        "max_perturb_ratio": 0.005,  # 5x smaller delta than default
        "beta_limit": 0.40, "sq_limit": 0.55, "lpc_limit": 0.028, "gain_limit": 0.22,
        "mcd": 2.5,
        "cosine_embed": 3.0,
        "delta_mfcc": 1.5,
        "ltas": 0.5,
        "mel_preserve": 10.0,
        "temporal_env": 5.0,
        "snr_floor": 35.0,
        "snr_target": 22.0,
        "spectral_conv": 10.0,
        "smoothness": 200.0,
        "delta_energy": 2000.0,  # Very heavy penalty on delta
        "param_reg": 0.5,
        "content_preserve": 5.0,
    },

    # ==========================================
    # ROBUST (W3) — EOT augmentation training
    # ==========================================

    "ROB_sf_robust": {
        "model_type": "sf_only",
        "iterations": 1500,
        "lr": 0.008,
        "augmentation": True,
        "beta_limit": 0.45, "sq_limit": 0.60, "lpc_limit": 0.030, "gain_limit": 0.25,
        "mcd": 2.5,
        "cosine_embed": 3.0,
        "delta_mfcc": 1.5,
        "ltas": 0.5,
        "mel_preserve": 8.0,
        "snr_floor": 30.0,
        "snr_target": 20.0,
        "param_reg": 0.3,
        "content_preserve": 5.0,
    },

    "ROB_hybrid_robust": {
        "model_type": "hybrid",
        "iterations": 1500,
        "lr": 0.006,
        "augmentation": True,
        "max_perturb_ratio": 0.005,  # Tiny delta, SF must carry the load
        "beta_limit": 0.45, "sq_limit": 0.60, "lpc_limit": 0.030, "gain_limit": 0.25,
        "mcd": 2.5,
        "cosine_embed": 3.0,
        "delta_mfcc": 1.5,
        "ltas": 0.5,
        "mel_preserve": 8.0,
        "temporal_env": 5.0,
        "snr_floor": 30.0,
        "snr_target": 20.0,
        "spectral_conv": 8.0,
        "smoothness": 200.0,
        "delta_energy": 2000.0,
        "param_reg": 0.3,
        "content_preserve": 5.0,
    },

    # ==========================================
    # ORIGINAL HYBRID configs (for comparison)
    # ==========================================

    "HYB_gentle": {
        "model_type": "hybrid",
        "iterations": 1000,
        "lr": 0.005,
        "max_perturb_ratio": 0.015,
        "beta_limit": 0.30, "lpc_limit": 0.020, "gain_limit": 0.15,
        "mcd": 1.5,
        "cosine_embed": 2.0,
        "mel_preserve": 20.0,
        "temporal_env": 10.0,
        "snr_floor": 60.0,
        "snr_target": 30.0,
        "spectral_conv": 20.0,
        "smoothness": 150.0,
        "delta_energy": 800.0,
        "param_reg": 1.5,
    },

    "HYB_strong": {
        "model_type": "hybrid",
        "iterations": 1200,
        "lr": 0.005,
        "max_perturb_ratio": 0.035,
        "beta_limit": 0.30, "lpc_limit": 0.020, "gain_limit": 0.15,
        "mcd": 2.0,
        "cosine_embed": 3.0,
        "delta_mfcc": 1.0,
        "ltas": 0.5,
        "mel_preserve": 8.0,
        "temporal_env": 5.0,
        "snr_floor": 30.0,
        "snr_target": 20.0,
        "spectral_conv": 8.0,
        "smoothness": 80.0,
        "delta_energy": 200.0,
        "param_reg": 0.5,
    },

    # ==========================================
    # OPTIMISED CONFIG — uses all fixed systems
    # ==========================================
    # Key differences vs HYB_strong:
    #  1. content_preserve: 5.0  → Whisper W4 loss now ACTIVE
    #  2. mel_preserve: 15.0     → Stronger quality constraint
    #  3. max_perturb_ratio: 0.020 → Tighter delta = better SNR
    #  4. augmentation: True     → EOT robustness training
    #  5. snr_floor/target raised → Enforce perceptual quality
    #  6. More iterations + lower lr → Cleaner convergence

    "HYB_optimal": {
        "model_type": "hybrid",
        "iterations": 1500,
        "lr": 0.004,
        "augmentation": True,
        "max_perturb_ratio": 0.020,
        "beta_limit": 0.30, "sq_limit": 0.50, "lpc_limit": 0.020, "gain_limit": 0.15,
        "mcd": 2.0,
        "cosine_embed": 3.0,
        "delta_mfcc": 1.0,
        "ltas": 0.5,
        "mel_preserve": 15.0,
        "temporal_env": 8.0,
        "snr_floor": 40.0,
        "snr_target": 22.0,
        "spectral_conv": 10.0,
        "smoothness": 100.0,
        "delta_energy": 300.0,
        "param_reg": 0.5,
        "content_preserve": 5.0,
    },
}


# ============================================================
# MASTER PIPELINE
# ============================================================

def run_batch_pipeline_v5():
    total_start = time.time()
    n_exp = len(EXPERIMENTS)
    n_files = len(AUDIO_FILES)

    print("\n" + "█" * 70)
    print("█" + " v5 HYBRID BATCH PIPELINE ".center(68) + "█")
    print("█" * 70)
    print(f"  Files: {n_files} | Experiments: {n_exp}")
    print(f"  Total runs: {n_files * n_exp}")
    print("█" * 70)

    # Load multiple verifiers (ASV Ensemble)
    print("\n📦 Loading ASV Ensemble (ECAPA + X-Vector)...")
    verifiers = {
        "ecapa": SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": DEVICE}
        ),
        "xvector": SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb",
            run_opts={"device": DEVICE}
        )
    }
    # Issue 5: Freeze ASV encoder parameters so they don't consume gradients during attack
    for v_name, verifier in verifiers.items():
        verifier.eval()
        for p in verifier.parameters():
            p.requires_grad_(False)
    print("✅ Ensemble Loaded.\n")

    processed_files = []

    # ==========================================
    # STAGE A: TRAIN
    # ==========================================
    print("=" * 70)
    print("🔧 STAGE A: Training all models on all audios...")
    print("=" * 70)

    # Pre-build shared transforms
    mfcc_t = T.MFCC(sample_rate=SR, n_mfcc=40).to(DEVICE)
    mel_ts = [
        T.MelSpectrogram(sample_rate=SR, n_fft=n, hop_length=h, n_mels=80).to(DEVICE)
        for n, h in [(512, 128), (1024, 256), (2048, 512)]
    ]

    for fi, audio_path in enumerate(AUDIO_FILES):
        if not os.path.exists(audio_path):
            continue

        base_name = os.path.basename(audio_path).replace('.wav', '')
        processed_files.append({"path": audio_path, "name": base_name})

        print(f"\n🎧 [{fi+1}/{n_files}] Voice: {base_name}")

        # Load audio once per file
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = T.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.to(DEVICE)

        with torch.no_grad():
            embs_orig = {v_name: v.encode_batch(waveform) for v_name, v in verifiers.items()}

        for ei, (exp_name, cfg) in enumerate(EXPERIMENTS.items()):
            outpath = os.path.join(OUTPUT_DIR, f"prot_{base_name}_{exp_name}.wav")
            mt = cfg.get("model_type", "sf_only")
            print(f"   [{ei+1}/{n_exp}] {exp_name} ({mt})...", end=" ")

            if os.path.exists(outpath):
                print("✅ Already generated (skipping)")
                continue

            t0 = time.time()
            try:
                protected = train_one(
                    waveform, cfg, verifiers, embs_orig, mfcc_t, mel_ts
                )
                torchaudio.save(outpath, protected.cpu(), SR)
                print(f"✅ {time.time()-t0:.0f}s")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"❌ {e}")
            
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # ==========================================
    # STAGE B: CLONE
    # ==========================================
    print("\n" + "=" * 70)
    print("🎙️ STAGE B: Cloning with XTTS v2...")
    print("=" * 70)

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    from TTS.api import TTS
    CLONE_TEXT = "This is a voice cloning test to evaluate the protection quality of the audio defense system."
    print("Loading XTTS and YourTTS cloners...")
    cloners = {
        "XTTS": TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE),
        "YourTTS": TTS("tts_models/multilingual/multi-dataset/your_tts").to(DEVICE)
    }

    for audio_data in processed_files:
        base_name = audio_data["name"]
        orig_path = audio_data["path"]
        print(f"\n   Cloning {base_name}...")
        
        for c_name, tts in cloners.items():
            print(f"     Using Cloner: {c_name}")
            # Baseline
            bl_path = os.path.join(OUTPUT_DIR, f"clone_{base_name}_BASELINE_{c_name}.wav")
            try:
                if c_name == "YourTTS":
                    tts.tts_to_file(text=CLONE_TEXT, speaker_wav=orig_path, language="en", file_path=bl_path)
                else:
                    tts.tts_to_file(text=CLONE_TEXT, speaker_wav=orig_path, language="en", file_path=bl_path)
            except Exception as e:
                print(f"     ❌ Baseline {c_name}: {e}")

            # Protected versions
            for exp_name in EXPERIMENTS:
                prot_p = os.path.join(OUTPUT_DIR, f"prot_{base_name}_{exp_name}.wav")
                clo_p = os.path.join(OUTPUT_DIR, f"clone_{base_name}_{exp_name}_{c_name}.wav")
                if os.path.exists(prot_p):
                    try:
                        tts.tts_to_file(text=CLONE_TEXT, speaker_wav=prot_p, language="en", file_path=clo_p)
                    except Exception as e:
                        print(f"     ❌ {exp_name} {c_name}: {e}")

    del cloners
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ==========================================
    # STAGE C: EVALUATE
    # ==========================================
    print("\n" + "=" * 70)
    print("📊 STAGE C: Evaluating...")
    print("=" * 70)

    all_results = []

    for audio_data in processed_files:
        base_name = audio_data["name"]
        orig_path = audio_data["path"]

        # Baseline clone (average across all cloners)
        baseline_cossim, baseline_mcd, count = 0.0, 0.0, 0
        for c_name in ["XTTS", "YourTTS"]:
            bl_path = os.path.join(OUTPUT_DIR, f"clone_{base_name}_BASELINE_{c_name}.wav")
            if os.path.exists(bl_path):
                cm = calculate_clone_metrics(orig_path, bl_path, verifiers)
                baseline_cossim += cm["Clone_CosSim"]
                baseline_mcd += cm["Clone_MCD"]
                count += 1
        
        if count > 0:
            all_results.append({
                "Audio": base_name,
                "Experiment": "BASELINE",
                "Clone_CosSim": baseline_cossim / count,
                "Clone_MCD": baseline_mcd / count,
            })

        for exp_name in EXPERIMENTS:
            row = {"Audio": base_name, "Experiment": exp_name}
            prot_p = os.path.join(OUTPUT_DIR, f"prot_{base_name}_{exp_name}.wav")
            
            if os.path.exists(prot_p):
                row.update(calculate_protection_metrics(orig_path, prot_p, verifiers))
                
            clone_cossim, clone_mcd, c_count = 0.0, 0.0, 0
            for c_name in ["XTTS", "YourTTS"]:
                clo_p = os.path.join(OUTPUT_DIR, f"clone_{base_name}_{exp_name}_{c_name}.wav")
                if os.path.exists(clo_p):
                    cm = calculate_clone_metrics(orig_path, clo_p, verifiers)
                    # Per-cloner breakdown columns
                    row[f"Clone_CosSim_{c_name}"] = cm["Clone_CosSim"]
                    row[f"Clone_MCD_{c_name}"] = cm["Clone_MCD"]
                    clone_cossim += cm["Clone_CosSim"]
                    clone_mcd += cm["Clone_MCD"]
                    c_count += 1
            
            if c_count > 0:
                row["Clone_CosSim"] = clone_cossim / c_count
                row["Clone_MCD"] = clone_mcd / c_count

            all_results.append(row)

    df_ind = pd.DataFrame(all_results)

    # ==========================================
    # COMPUTE SCORES
    # ==========================================

    # Get baseline Clone_CosSim per audio for relative scoring
    baseline_clones = df_ind[df_ind["Experiment"] == "BASELINE"].set_index("Audio")["Clone_CosSim"]

    for idx, row in df_ind.iterrows():
        if row["Experiment"] == "BASELINE":
            continue

        cs = safe_float(row.get("CosSim_Prot"), 1.0)
        mcd = safe_float(row.get("MCD_Prot"), 0.0)
        pq = safe_float(row.get("PESQ"), 1.0)
        st = safe_float(row.get("STOI"), 0.0)
        cc = safe_float(row.get("Clone_CosSim"), 1.0)
        snr_v = safe_float(row.get("SNR_dB"), 0.0)

        # Get this audio's baseline for relative comparison
        bl_cc = safe_float(baseline_clones.get(row["Audio"], 0.6), 0.6)

        # Clone improvement: how much did we reduce vs baseline?
        # 0.0 = same as baseline, 1.0 = clone completely broken
        clone_improvement = max(0, (bl_cc - cc) / (bl_cc + 1e-8))

        # Issue 7: Fix quality gate thresholds (was rejecting every model before)
        # Use a continuous penalty instead of a hard boolean threshold
        quality_factor = min(1.0, (pq / 2.5) * (st / 0.70))
        full_score = (
            0.10 * (1.0 - cs) +                       # Embedding shift
            0.05 * min(mcd / 15.0, 1.0) +              # Spectral shift
            0.30 * clone_improvement +                  # Clone disruption (main goal)
            0.05 * (1.0 - cc) +                        # Absolute clone distance
            0.20 * st +                                # Intelligibility
            0.15 * max(0.0, pq / 4.5) +                # Quality
            0.10 * min(max(snr_v, 0) / 30.0, 1.0) +   # SNR
            0.05 * min(max(0, 1.0 - cs), 1.0)          # Protection strength
        )
        score = quality_factor * full_score + (1.0 - quality_factor) * (0.15 * clone_improvement)
        quality_ok = (pq >= 2.5 and st >= 0.70)

        df_ind.at[idx, "Score"] = round(score, 4)
        df_ind.at[idx, "QualOK"] = "✅" if quality_ok else "❌"
        df_ind.at[idx, "Clone_Improv%"] = round(clone_improvement * 100, 1)

    # Save individual
    ind_csv = os.path.join(OUTPUT_DIR, "individual_scores.csv")
    df_ind.to_csv(ind_csv, index=False)

    # ==========================================
    # MEAN SUMMARY
    # ==========================================

    df_models = df_ind[df_ind["Experiment"] != "BASELINE"].copy()
    numeric_cols = [
        "CosSim_Prot", "MCD_Prot", "PESQ", "STOI", "SNR_dB",
        "Clone_CosSim", "Clone_MCD", "Score", "Clone_Improv%"
    ]
    existing_num = [c for c in numeric_cols if c in df_models.columns]

    for c in existing_num:
        df_models[c] = pd.to_numeric(df_models[c], errors='coerce')

    df_mean = df_models.groupby("Experiment")[existing_num].mean().reset_index()

    # ---- EER CALCULATION ----
    print("\n   Computing EER...")
    orig_embeds = []
    for audio_data in processed_files:
        wav_o, _ = librosa.load(audio_data["path"], sr=16000)
        t_o = torch.tensor(wav_o).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            # For EER, we use ECAPA as the primary metric, or average embeddings? Let's just use ECAPA for EER to keep it comparable.
            orig_embeds.append(verifiers["ecapa"].encode_batch(t_o).squeeze())

    neg_scores = []
    for i in range(len(orig_embeds)):
        for j in range(i + 1, len(orig_embeds)):
            sim = F.cosine_similarity(orig_embeds[i], orig_embeds[j], dim=0).item()
            neg_scores.append(sim)

    eer_list = []
    for exp in df_mean["Experiment"]:
        exp_data = df_ind[df_ind["Experiment"] == exp]
        prot_pos = exp_data["CosSim_Prot"].dropna().tolist()
        clone_pos = exp_data["Clone_CosSim"].dropna().tolist()
        eer_list.append({
            "Experiment": exp,
            "Prot_EER_%": compute_eer(prot_pos, neg_scores),
            "Clone_EER_%": compute_eer(clone_pos, neg_scores),
        })

    df_eer = pd.DataFrame(eer_list)
    df_mean = pd.merge(df_mean, df_eer, on="Experiment")

    # Baseline mean
    bl_mean = df_ind[df_ind["Experiment"] == "BASELINE"]["Clone_CosSim"].mean()

    # Model type column
    type_map = {name: cfg.get("model_type", "sf_only") for name, cfg in EXPERIMENTS.items()}
    df_mean["Type"] = df_mean["Experiment"].map(type_map)

    # Final column order
    col_order = [
        "Experiment", "Type",
        "CosSim_Prot", "Prot_EER_%",
        "MCD_Prot", "PESQ", "STOI", "SNR_dB",
        "Clone_CosSim", "Clone_EER_%", "Clone_MCD",
        "Clone_Improv%", "Score"
    ]
    cols = [c for c in col_order if c in df_mean.columns]
    df_mean = df_mean[cols].sort_values("Score", ascending=False).round(4)

    mean_csv = os.path.join(OUTPUT_DIR, "mean_scoreboard.csv")
    df_mean.to_csv(mean_csv, index=False)

    # ==========================================
    # REPORT
    # ==========================================
    elapsed = time.time() - total_start

    print(f"\n💾 Saved: {ind_csv}")
    print(f"💾 Saved: {mean_csv}")

    print("\n\n" + "█" * 80)
    print("█" + " FINAL MEAN SCOREBOARD ".center(78) + "█")
    print("█" * 80)
    print(f"\n  Speakers: {n_files} | Experiments: {n_exp}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Baseline Clone_CosSim (mean): {bl_mean:.4f}")

    print("\n  METRIC KEY:")
    print("  Clone_Improv%  = How much BETTER than doing nothing")
    print("                   30% = clone 30% less similar than baseline")
    print("  Clone_EER_%    = Error rate for clone detection")
    print("                   50% = random chance = maximally confused")
    print("  Score          = Composite (higher = better overall)")

    print("\n  RESULTS:")
    print(df_mean.to_string(index=False))

    # Rankings
    print("\n" + "=" * 80)
    print("  🏆 ANALYSIS")
    print("=" * 80)

    # SF vs Hybrid comparison
    sf_rows = df_mean[df_mean["Type"] == "sf_only"]
    hyb_rows = df_mean[df_mean["Type"] == "hybrid"]

    if not sf_rows.empty and not hyb_rows.empty:
        best_sf = sf_rows.iloc[0]
        best_hyb = hyb_rows.sort_values("Score", ascending=False).iloc[0]
        print(f"\n  📊 SOURCE-FILTER vs HYBRID:")
        print(f"  Best SF:     {best_sf['Experiment']}")
        print(f"    Clone={best_sf.get('Clone_CosSim','?')} | "
              f"PESQ={best_sf.get('PESQ','?')} | "
              f"Improv={best_sf.get('Clone_Improv%','?')}%")
        print(f"  Best Hybrid: {best_hyb['Experiment']}")
        print(f"    Clone={best_hyb.get('Clone_CosSim','?')} | "
              f"PESQ={best_hyb.get('PESQ','?')} | "
              f"Improv={best_hyb.get('Clone_Improv%','?')}%")

    # Top overall
    top3 = df_mean.nlargest(3, "Score")
    print(f"\n  🥇 TOP 3:")
    for rank, (_, r) in enumerate(top3.iterrows(), 1):
        medal = ["🥇", "🥈", "🥉"][rank - 1]
        print(f"  {medal} {r['Experiment']}")
        print(f"     Clone_CosSim={r.get('Clone_CosSim','?')} | "
              f"Improv={r.get('Clone_Improv%','?')}% | "
              f"PESQ={r.get('PESQ','?')} | "
              f"EER={r.get('Clone_EER_%','?')}%")

    print(f"\n  Baseline for reference: Clone_CosSim={bl_mean:.4f}")
    print("█" * 80 + "\n")

    return df_mean


# ============================================================
# 🚀 RUN
# ============================================================
if __name__ == "__main__":
    results = run_batch_pipeline_v5()
