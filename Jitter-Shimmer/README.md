# Audio Medusa: Jitter–Shimmer Protection Research

This repository contains the **jitter and shimmer protection experiments** developed as part of the Audio Medusa audio-deepfake defense project.

The goal is to make a speaker's voice harder to clone by modifying fine-grained vocal characteristics while preserving speech intelligibility and acceptable perceptual quality.

---

## Project Motivation

Modern voice-cloning systems learn speaker identity from multiple acoustic cues, including:

- Pitch and prosody
- Harmonic and spectral structure
- Formants and vocal-tract characteristics
- Glottal excitation behavior
- Amplitude variation
- Timing and phase-related information

This repository focuses on two subtle but speaker-relevant voice characteristics:

- **Jitter:** Small cycle-to-cycle variation in vocal-fold timing or pitch period.
- **Shimmer:** Small cycle-to-cycle variation in vocal amplitude.

The central idea is to introduce controlled changes to these micro-variations so that speaker-specific information becomes less stable for voice-cloning models.

---

## Research Question

> Can jitter and shimmer perturbations reduce speaker identity similarity without making the protected speech unusable for human listeners?

This is a difficult trade-off.

- Very weak perturbations preserve quality but can be ignored by robust cloning systems.
- Very strong perturbations may reduce similarity but can create clicks, distortion, robotic artifacts, or poor intelligibility.

---

## Implementations

This repository preserves multiple implementation styles rather than presenting a single method as universally successful.

| Folder | Implementation | Purpose |
|---|---|---|
| [`01_cycle_level_baseline`](./01_cycle_level_baseline) | Explicit cycle-level jitter/shimmer editing | Classical baseline; useful for understanding why direct waveform-cycle manipulation is unstable |
| [`02_jswarp`](./02_jswarp) | Differentiable smooth jitter–shimmer warping | Main learned approach; predicts continuous timing and amplitude fields |
| [`03_stft_audio_protector`](./03_stft_audio_protector) | STFT-domain multi-component perturbation | Broader Audio Medusa implementation combining jitter, shimmer, phase, and excitation perturbations |
| [`04_xtts_encoder_guided`](./04_xtts_encoder_guided) | XTTS speaker-encoder-guided optimization | Attack-aware speaker-identity protection and evaluation direction |
| [`evaluation`](./evaluation) | Shared evaluation protocol | Measures protection, audio quality, and cloning robustness |

---

## Evolution of the Project

```text
Cycle-level waveform editing
            │
            ▼
Poor quality: clicks, phase breaks, damaged harmonics
            │
            ▼
Smooth differentiable JSWarp fields
            │
            ▼
Better quality, but weak protection against XTTS
            │
            ▼
Stronger speaker-embedding objectives
            │
            ▼
STFT multi-component perturbation and XTTS-guided optimization
```

---

## Core Lessons

### Smooth perturbations are safer than hard edits

Directly cutting, stretching, and stitching waveform cycles can change jitter and shimmer metrics, but it often damages phase continuity and harmonic structure.

### Jitter and shimmer alone are not enough

Modern voice-cloning systems can remain robust to mild prosodic changes if speaker-defining spectral and formant information is still preserved.

### Loss alignment matters

A mel-spectrogram proxy may look successful during training while failing to reduce real speaker similarity during evaluation.

The identity loss should use a speaker encoder that is meaningful for the intended threat model.

### Protection and quality must be reported separately

A good result should not only increase jitter or shimmer. It should also show:

- Reduced speaker similarity
- Reduced cloning similarity
- Preserved intelligibility
- Acceptable audio quality
- No severe waveform artifacts

---

## Evaluation Dimensions

| Category | Example Measurements |
|---|---|
| Jitter–shimmer change | Relative jitter and shimmer values before and after protection |
| Speaker protection | Original-to-protected embedding cosine similarity |
| Clone protection | Similarity between original voice and clone generated from protected audio |
| Audio quality | PESQ, STOI, SNR, mel-spectrogram distance |
| Signal stability | Clicks, discontinuities, clipping, waveform and spectrogram inspection |
| Perception | Human listening comparison between original and protected audio |

---
