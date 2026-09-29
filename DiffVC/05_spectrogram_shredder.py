# ==========================================
# MODULE 1: SETUP, MODELS, AND BASELINE
# ==========================================
import os
import json
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import IPython.display as ipd
from pathlib import Path
import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
import inspect

# --- BULLETPROOF LIBROSA MONKEYPATCH (Anti-Recursion) ---
if not hasattr(librosa.feature.melspectrogram, '_is_patched'):
    _orig_melspectrogram = librosa.feature.melspectrogram
    def _patched_melspectrogram(*args, **kwargs):
        if len(args) >= 2: kwargs['y'], kwargs['sr'] = args[0], args[1]; args = args[2:]
        elif len(args) == 1: kwargs['y'] = args[0]; args = tuple()
        return _orig_melspectrogram(*args, **kwargs)
    _patched_melspectrogram._is_patched = True
    librosa.feature.melspectrogram = _patched_melspectrogram

if not hasattr(librosa.resample, '_is_patched'):
    _orig_resample = librosa.resample
    def _patched_resample(*args, **kwargs):
        if len(args) >= 3: kwargs['y'], kwargs['orig_sr'], kwargs['target_sr'] = args[0], args[1], args[2]; args = args[3:]
        return _orig_resample(*args, **kwargs)
    _patched_resample._is_patched = True
    librosa.resample = _patched_resample

os.chdir('/content/Speech-Backbones/DiffVC')

# --- IMPORT REPO MODULES ---
import params
from model import DiffVC
import sys
sys.path.append('hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder

use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'

# --- DIFFERENTIABLE PREPROCESSING ---
class DifferentiableMel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_basis = torch.from_numpy(librosa_mel_fn(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)).float().to(device)

    def forward(self, wav):
        wav = F.pad(wav, (384, 384), mode='reflect')
        stft = torch.stft(wav, n_fft=1024, hop_length=256, win_length=1024,
                          window=torch.hann_window(1024).to(device), center=False, return_complex=True)
        stftm = torch.sqrt(torch.real(stft)**2 + torch.imag(stft)**2 + 1e-9)
        mel = torch.matmul(self.mel_basis, stftm)
        return torch.log(torch.clamp(mel, min=1e-5))

# --- LOAD MODELS ---
def load_models():
    print("Loading Models into Memory...")
    gen = DiffVC(params.n_mels, params.channels, params.filters, params.heads, params.layers, params.kernel,
                 params.dropout, params.window_size, params.enc_dim, params.spk_dim, params.use_ref_t,
                 params.dec_dim, params.beta_min, params.beta_max).to(device)
    gen.load_state_dict(torch.load('checkpts/vc/vc_libritts_wodyn.pt', map_location=device, weights_only=False))
    gen.eval()

    with open('checkpts/vocoder/config.json') as f: h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h).to(device)
    vocoder.load_state_dict(torch.load('checkpts/vocoder/generator', map_location=device, weights_only=False)['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()

    spk_encoder.load_model(Path('checkpts/spk_encoder/pretrained.pt'), device=device)
    return gen, vocoder

generator, vocoder = load_models()
diff_mel = DifferentiableMel()

# --- HIGH-FIDELITY INFERENCE PIPELINE ---
def get_official_mel(wav_path_or_tensor):
    if isinstance(wav_path_or_tensor, str):
        wav, _ = load(wav_path_or_tensor, sr=22050)
    else:
        wav = wav_path_or_tensor.detach().cpu().numpy().squeeze()

    mel = librosa.feature.melspectrogram(y=wav, sr=22050, n_fft=1024, hop_length=256,
                                         win_length=1024, n_mels=80, fmin=0, fmax=8000)
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    return torch.from_numpy(mel).float().unsqueeze(0).to(device)

def quick_convert(source_wav, target_path):
    m_src = get_official_mel(source_wav)
    m_src_len = torch.LongTensor([m_src.shape[-1]]).to(device)

    m_tgt = get_official_mel(target_path)
    m_tgt_len = torch.LongTensor([m_tgt.shape[-1]]).to(device)

    embed_tgt = torch.from_numpy(spk_encoder.embed_utterance(spk_encoder.preprocess_wav(target_path))).float().unsqueeze(0).to(device)

    with torch.no_grad():
        _, mel_out = generator.forward(m_src, m_src_len, m_tgt, m_tgt_len, embed_tgt, n_timesteps=30, mode='ml')
        audio = vocoder.forward(mel_out).cpu().squeeze().clamp(-1, 1)
    return audio

# --- INITIALIZE ENCODER ---
_ = torch.from_numpy(spk_encoder.preprocess_wav('example/6415_111615_000012_000005.wav')).float().to(device)
print("✅ Baseline models initialized and ready for defense protocols.")


# ==========================================
# MODULE 2: SPECTROGRAM SHREDDER (UNBREAKABLE GRAPH)
# ==========================================

def run_spectrogram_shredder(src_path, tgt_path, iterations=300, epsilon=0.004, lr=0.0005):
    print(f"Executing Spectrogram Shredder... (Iters: {iterations}, Eps: {epsilon})")

    # 1. Setup Source Audio
    wav, _ = load(src_path, sr=22050)
    wav = torch.from_numpy(wav).float().unsqueeze(0).to(device)

    delta = torch.zeros_like(wav, requires_grad=True)

    # 2. Perceptual Masking
    rms = torch.sqrt(F.avg_pool1d(wav**2, kernel_size=1024, stride=1, padding=512) + 1e-8)[:, :wav.shape[-1]]
    mask = rms / (rms.mean() * 1.5 + 1e-8)
    mask = torch.clamp(mask, min=0.4, max=1.0)

    # 3. Create the Lethal Target
    with torch.no_grad():
        mel_clean = diff_mel(wav)
        mel_target = mel_clean.flip(dims=[1])

    print("Initiating direct frequency distortion...")

    # 4. The Attack Loop
    for i in range(iterations):
        if delta.grad is not None:
            delta.grad.zero_()

        perturbed_wav = wav + (delta * mask)
        mel_adv = diff_mel(perturbed_wav)

        # Match our upside-down target
        loss = F.l1_loss(mel_adv, mel_target)
        loss.backward()

        assert delta.grad is not None, "Graph broke!"

        with torch.no_grad():
            delta.sub_(lr * delta.grad.sign())
            delta.clamp_(-epsilon, epsilon)

        if (i + 1) % 50 == 0:
            print(f" Iteration {i+1:3d}/{iterations} | Spectrogram Inversion Loss: {loss.item():.4f}")

    poisoned_tensor = (wav + (delta * mask)).detach()
    perturbation_mag = (delta * mask).abs().max().item()
    print(f"\n✅ Attack complete! Max applied perturbation: {perturbation_mag:.6f}")

    return poisoned_tensor

# ==========================================
# EXECUTION & PROTECTION TEST
# ==========================================
tgt_path = 'example/8534_216567_000015_000010.wav'
src_path = 'example/6415_111615_000012_000005.wav'

# Tuned parameters for stealth and stability
poisoned_audio_tensor = run_spectrogram_shredder(
    src_path,
    tgt_path,
    epsilon=0.02,   # True stealth boundary
    lr=0.0005,       # Smooth mathematical stepping
    iterations=500
)

print("\n" + "="*60)
print("RESULTS")
print("="*60)

normal_clone = quick_convert(src_path, tgt_path)
attacked_clone = quick_convert(poisoned_audio_tensor, tgt_path)

print("1. ORIGINAL SOURCE (Unprotected)")
ipd.display(ipd.Audio(src_path))

print("\n2. SECURED SOURCE (Waveform Watermark Applied - Should sound clean!)")
ipd.display(ipd.Audio(poisoned_audio_tensor.cpu().numpy().squeeze(), rate=22050))

print("\n3. NORMAL CLONE (Clean trajectory)")
ipd.display(ipd.Audio(normal_clone, rate=22050))

print("\n4. BLOCKED CLONE (SDE processing gibberish features)")
ipd.display(ipd.Audio(attacked_clone, rate=22050))
