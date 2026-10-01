Source Filter Model
Overview
The Glottal Source (E): The raw excitation produced by the vibrating vocal cords. This is a quasi-periodic pulse train whose shape encodes features like breathiness, creakiness, and voice tension.
The Vocal Tract Filter (A): The resonant cavity formed by the throat, mouth, and nasal passage. Its shape changes with articulation (vowels, consonants) and is mathematically captured as an all-pole filter A(z)

Linear Predictive Coding (LPC)

LPC is the standard technique for estimating A(z). The principle is that any sample of speech can be predicted as a linear combination of the previous p samples, where p is the model order. Solving the Yule-Walker equations via autocorrelation yields the p filter coefficients that best model the vocal tract for a given audio frame.
Classical LPC implementations are non-differentiable, making them incompatible with gradient-based optimization. Medusa implements the entire pipeline using PyTorch operations (FFT, torch.linalg.solve, F.conv1d), which preserves the computation graph through every step. This unlocks end-to-end training: a loss function defined on the output audio can propagate gradients all the way back through re-synthesis, through the perturbation heads, through inverse filtering, and into the upstream encoder — enabling joint optimization of the entire pipeline.
Module / Phase
Class Name
Key Parameter
Role in Architecture
Phase 1
DifferentiableSourceFilter
order=16
Decomposes waveform → (A-coeffs, Source E)
Phase 2
SpectralTiltAttacker
β (learnable)
Modifies vocal tract tilt via 1 − βz⁻¹ filter
Phase 3
LF_SpeedQuotientAttacker
sq_shift (learnable)
Alters glottal pulse asymmetry via phase warping


Phase 1 — Differentiable Source-Filter Decoupler
Step 1 — Autocorrelation via FFT
n_fft = 2 ** ceil(log2(frame_len * 2 - 1))  # next power of 2
X = rfft(x, n=n_fft)                   # frequency domain
power   = |X|²                               # power spectrum
r  = irfft(power, n=n_fft)[:order+1]    # autocorrelation lags
Step 2 — Toeplitz Matrix Construction
The Yule-Walker equations for LPC are expressed as a symmetric Toeplitz linear system R·a = p, where R[i,j] = r[|i−j|]. The module constructs this batch-wise in a nested loop — acceptable at inference time for typical LPC orders (16), as the matrix is only 16×16.
Step 3 — Solving R·a = p (Differentiably)
Step 4 — Glottal Inverse Filtering
a_coeffs = [1, -a₁, -a₂, ..., -a_p]       # A(z) polynomial
residual  = conv1d(x, flip(a_coeffs))       # Apply inverse filter
source_E  = residual[:, :, :-order]         # Trim padding artifact


Phase 2 — Spectral Tilt Attacker
The 1 − βz⁻¹ Filter
The perturbation applied is a first-order FIR filter of the form H(z) = 1 − βz⁻¹. In the discrete-time frequency domain:
β > 0  →  high-frequency boost  (tilts spectrum upward → tenser, pressed quality)
β < 0  →  high-frequency cut    (tilts spectrum downward → breathier quality)
β = 0  →  identity (no change)
β is initialized at 0.0 and wrapped as nn.Parameter. The Adam optimizer can adjust it during training. Because the filter is applied via F.conv1d, the gradient ∂(output)/∂β is well-defined and non-zero, making this parameter trainable.

Phase 3 — LF Speed Quotient Attacker
The Speed Quotient (SQ) is an LF glottal model parameter that describes the asymmetry of each glottal pulse: how quickly the vocal folds open versus how slowly they close. A higher SQ corresponds to a more abrupt closure (modal/pressed voice); a lower SQ to a gentler closure (breathy/soft onset).
Phase Manipulation in the Frequency Domain
Asymmetry in the time domain corresponds to phase non-linearity in the frequency domain. To shift the SQ without altering the magnitude spectrum (i.e., without changing the overall timbre or energy), this module operates exclusively on the phase of the source excitation E.
E_freq   = rfft(source_excitation)             # complex spectrum
magnitude = |E_freq|                           # preserved exactly
phase     = angle(E_freq)                      # modified
freq_bins = linspace(0, π, num_bins)           # normalized frequencies
dispersion = sq_shift × sin(freq_bins)         # non-linear warp curve
new_phase = phase + dispersion                 # apply warp
E_warped  = magnitude × exp(j × new_phase)    # reconstruct
warped_source = irfft(E_warped)                # back to time domain
Experiment 1 — Loss Function Sweep on 2-Parameter Model
What changed from baseline: The three separate modules were merged into a single MasterBiometricAttacker. Re-synthesis was added via frequency-domain division (E_warped / A_freq). A full automated training pipeline tested 16 loss function combinations across 4 groups — single losses, pairs, full combinations, and weight variations. An ECAPA-TDNN verifier (SpeechBrain) was loaded for embedding-based evaluation. XTTS v2 was used to clone protected audio and measure clone quality.₹
Parameter
Value
Learnable parameters
2 (β, sq_shift)
Experiments run
16 combinations across 4 groups
Clamps
β ±0.98, sq_shift ±3.14
Iterations
600 per experiment
Audio files
1 (sample1.wav)
Verifier
ECAPA-TDNN (SpeechBrain spkrec-ecapa-voxceleb)
Clone model
XTTS v2


Loss Functions Tested
Attack losses: MCD, cosine embedding similarity, LTAS divergence, spectral tilt, delta-MFCC, subband centroid, formant energy ratio.
Preservation losses: Multi-resolution mel, temporal envelope, parameter regularization.
Metric
Baseline (no protection)
Best result (E01_mcd_only)
Change
Clone_CosSim
0.5979
0.4891
↓ −18% — real disruption
CosSim_Prot
—
0.9867
Near-identical to original
PESQ
—
4.55
Excellent audio quality
STOI
—
0.9972
Fully intelligible
SNR_dB
—
−0.5 dB
Perturbation louder than signal
β convergence
0.0
0.98 (clamped)
Hit ceiling by step 100
sq_shift
0.0
0.0026
Negligible movement


Conclusion
The problem was not the loss functions. With only 2 parameters, the optimization landscape has one steep valley (β→boundary) and everything else is flat. Changing loss weights or combinations had no effect because the bottleneck was model capacity. The fix required more parameters, not better losses.
Experiment 2
What Changed from Experiment 1
Parameters expanded from 2 to 36 across four groups. Three new quality preservation losses were introduced. Clamps were made configurable with a systematic sweep. Gradient clipping was added at max_norm=2.0. A best-state checkpoint mechanism was introduced (saves model weights at the lowest loss step and restores before saving audio).
Parameter group
Count
Range (moderate)
What it controls
betas (tilt stages)
4
±0.35
Cascaded spectral tilt — 4th-order shaping
sq_coeffs (phase basis)
8
±0.50
8-harmonic glottal pulse warp
lpc_offsets (formant shift)
16
±0.025
Direct LPC coefficient perturbation
band_gains (EQ)
8
±0.20
8-band spectral equalizer
Total
36
—
vs 2 in Experiment 1


New quality losses: Waveform MSE (direct time-domain distance), SNR floor loss (one-sided: only activates when SNR drops below target), spectral convergence loss (Frobenius norm of magnitude spectrogram difference).
Metric
Exp 1 best
Exp 2 best (A3_moderate)
Change
Clone_CosSim
0.4891
0.4303
↓ Further improvement
PESQ
4.55
4.61
Maintained — excellent
STOI
0.9972
0.9985
Maintained
SNR_dB
−0.5 dB
+11.5 dB
↑ +12 dB — SNR floor loss worked
CosSim_Prot
0.9867
0.9970
Barely moved
Training stability
β always saturated
Stable — no saturation
Fixed by smaller param space + grad clip


Remaining Limitation
CosSim_Prot stayed at 0.997 across all quality-passing experiments. The ECAPA embedding space was not being disturbed. Clone disruption happened through XTTS's internal speaker encoder sensitivity — these are two different models measuring different things. The protection ceiling was approximately Clone_CosSim=0.43 with this architecture.
Experiment 3 V2
What Changed from Experiment 1
Same 36-parameter expansion as Experiment 2 but with the original loose clamps (β±0.85, LPC±0.15, EQ±2.0). This experiment was run in parallel with Experiment 2 before the tight-clamp boundary insight was fully applied.
Parameter
v1 clamp
v2 (loose) clamp
Effect of loose clamp
betas (tilt)
±0.98 (single)
±0.85 (each of 4)
4 cascaded = up to 4th-order shaping
sq_coeffs
±3.14 (single)
±1.50 (each of 8)
8-harmonic complex phase warping
lpc_offsets
not present
±0.15 (each of 16)
~300 Hz formant shift — audibly destructive
band_gains
not present
±2.0 (each of 8)
±17 dB per band — catastrophic

RESULTS
Config
CosSim_Prot
Clone_CosSim
PESQ
SNR_dB
Score
V1_mcd_baseline (control)
0.9867
0.5259
4.55
−0.5
0.6239
V2_A_mcd_only
0.6145
−0.0239
1.79
−2.3
0.7942
V2_B_mcd_dmfcc (best)
0.1806
0.1070
1.48
−5.6
0.8296
V2_B_mcd_cosine
0.5893
0.0417
1.70
−3.1
0.7791
V2_C_maxattack
0.0391
0.0391
1.15
−21.9
0.7183
V2_D_scheduled (worst)
−0.0322
−0.0153
1.12
−23.3
0.7200
V2_D_longrun
0.0389
0.0087
1.23
−16.1
0.8163


Key Observations
CosSim went negative in multiple experiments:  Negative cosine similarity means the clone embedding points in the opposite direction from the original in embedding space — complete identity destruction.
LPC offsets at ±0.15 are the audio-destruction source: Shifting formants by ~300 Hz produces audible artifacts. SNR ranged from −2 to −23 dB across all v2 experiments. In Experiment 2, tightening LPC to ±0.02 recovered SNR to +10 to +12 dB.
What This Experiment Confirmed
36 parameters with loose clamps can break voice identity far more aggressively than 2 parameters. The combination of Experiment 2 (tight clamps, good quality, weak protection) and Experiment 3 (loose clamps, strong protection, destroyed audio) together defined both ends of the fundamental tradeoff. 
Experiment 4 — Hybrid Architecture, v5 (10 Speakers)
What Changed from Experiment 3
Three simultaneous changes. First, a HybridAttacker class was introduced combining the SF module with a learnable raw waveform perturbation delta — a noise vector added to the SF output, constrained to a fraction of signal RMS (1.5%–4.5%). Second, evaluation moved from 1 to 10 LibriSpeech speakers, the first generalization test. Third, EER (Equal Error Rate) was added — 50% EER means the verifier is at random chance, i.e., maximally confused.
Config
max_perturb_ratio
Iterations
Key loss weights
CTRL_sf_moderate (control)
n/a — SF only
800
MCD 1.5, delta_mfcc 1.0, mel_preserve 10.0
HYB_gentle
1.5%
1000
MCD 1.5, cosine 2.0, snr_floor 60 (target 30dB)
HYB_moderate
2.5%
1200
MCD 2.0, cosine 2.5, delta_mfcc 0.8
HYB_strong
3.5%
1200
MCD 2.0, cosine 3.0, delta_mfcc 1.0, ltas 0.5
HYB_maximum
4.5%
1200
MCD 2.5, cosine 3.0, delta_mfcc 1.5, ltas 0.8

Results Across 10 Speakers (mean)
Config
Clone_CosSim
Clone_Improv%
PESQ
STOI
EER_Clone
Score
Baseline (no protection)
0.6478
—
—
—
—
—
CTRL_sf_moderate
0.5998
8.78%
4.357
0.9976
18.89%
0.4389
HYB_gentle
0.5178
19.05%
2.785
0.9839
18.89%
0.3997
HYB_moderate
0.5736
11.16%
2.100
0.9474
18.89%
0.1406
HYB_strong
0.5103
20.68%
1.577
0.9008
18.89%
0.0310
HYB_maximum
0.3501
46.14%
1.319
0.8166
29.44%
0.0692


Key Observations
Generalization confirmed: Trends held consistently across all 10 speakers, confirming the model is not overfitting to one speaker's LPC structure.
Tradeoff curve is clean and consistent: Going from gentle to maximum, Clone_CosSim dropped (0.52→0.35) and Clone_Improv% rose (19%→46%) proportionally. The delta perturbation magnitude scales protection predictably.
Conclusion
The delta perturbation works directionally and generalizes across speakers. But current regularization (smoothness, energy penalty, SNR floor) does not constrain the delta to be perceptually imperceptible. 
Complete Tradeoff Summary Across All Experiments
Experiment
Parameters
PESQ (best)
Clone_CosSim (best)
SNR_dB (best)
Key discovery
Baseline
2 (β, sq_shift)
Not tested
Not tested
Not tested
Architecture only — no training
Exp 1
2
4.55
0.4891
−0.5 dB
β always saturates — capacity bottleneck
Exp 2 (tight clamps)
36
4.61
0.4303
+11.5 dB
β±0.35 hard boundary — quality solved
Exp 3 (loose clamps)
36
1.48
0.1070
−5.6 dB
Attack power confirmed — quality destroyed
Exp 4 (hybrid, 10 spk)
36 + delta
2.785
0.3501
−0.12 dB
Generalizes — delta needs perceptual masking














Unsolved Issue/Improvements
Encoder and Cloner Mismatch Issue: 
The encoder the cloner is using could be anything, as the cloner is black box. So the integrated model should be focused on more encoder’s embedding space other than ECAPA(used here).
Psychoacoustic masking: 
Psychoacoustic masking thresholds, mel-domain constraints, or a neural perceptual loss (e.g. using a pretrained codec encoder as a perceptual proxy) would let the delta remain below the threshold of human perception while being large enough in embedding space to disrupt identity.
