# Medusa — Recommended Model Improvements

## 0. Executive Summary

Medusa's current architecture already has a strong research idea: perform speaker-protection perturbations in a differentiable source-filter representation rather than relying entirely on unconstrained waveform noise.

Current pipeline:

```text
waveform
   ↓
differentiable LPC/source-filter decomposition
   ↓
E (source) + A(z) (vocal tract)
   ↓
spectral tilt + SQ/phase + LPC + EQ perturbations
   ↓
optional raw waveform delta
   ↓
protected waveform
```

The main weakness is that the current strongest protection result is increasingly coming from the raw waveform perturbation, while the source-filter components are supposed to be the central contribution. The next version should therefore make the **structured source-filter pathway the primary mechanism**, and use the raw residual only as a tightly constrained finishing component.

The target for Medusa v2 should be:

> **A content-conditioned, source-filter-aware, transferable and purification-robust voice-protection model that maximizes cloning failure subject to strict perceptual and content constraints.**

---

# 1. Highest-Priority Changes

| Priority | Improvement | Why it matters |
|---|---|---|
| 1 | Multi-ASV + multi-TTS optimization | Removes dependence on ECAPA/one cloning model |
| 2 | Train through purification/adaptive transformations | Protects against modern removal attacks |
| 3 | Replace global 36 parameters with frame/content-conditioned parameters | Speech is highly nonstationary |
| 4 | Make structured source-filter perturbation primary; constrain raw delta | Prevents Medusa from becoming generic waveform noise |
| 5 | True glottal/LF parameterization | Makes the physiological claim much stronger |
| 6 | Stable LPC parameterization using reflection coefficients | Prevents destructive/unstable filter perturbations |
| 7 | Explicit speaker/content disentanglement | Prevents protection by simply damaging speech content |
| 8 | Psychoacoustic masking constraint | Directly targets imperceptibility |
| 9 | F0/harmonic-aware perturbation | Gives a more speaker-relevant signal representation |
| 10 | Codec/channel robustness | Makes protection survive real audio transformations |
| 11 | Universal + speaker-adaptive perturbation | Combines transferability with personalization |
| 12 | Temporal localization/masking | Concentrates perturbation where it is most effective |

---

# 2. Replace the Fixed 36-Parameter Model with a Learned Conditioning Network

## Current problem

The current model uses a relatively small set of global learned parameters:

```text
4 spectral-tilt parameters
8 SQ/phase parameters
16 LPC offsets
8 EQ gains
```

This is much better than the original 2-parameter model, but it is still essentially a global perturbation.

Speech is not stationary. Different frames contain different:

- vowels/consonants
- voiced/unvoiced regions
- pitch values
- formants
- phonetic information
- speaker-discriminative information.

## Recommended architecture

Instead of:

```text
theta = 36 global parameters
```

learn:

```text
theta_t = f_phi(E_t, A_t, F0_t, V_t, context_t)
```

where:

- `E_t` = local source representation
- `A_t` = local vocal-tract representation
- `F0_t` = pitch
- `V_t` = voicing information
- `context_t` = temporal context / learned speech representation.

The perturbation becomes:

```text
speech
  ↓
source-filter analysis
  ↓
frame representations
  ↓
conditioning network
  ↓
frame-specific perturbation parameters
```

This should let Medusa perturb different parts of speech differently.

## Expected advantage

A tiny perturbation applied only to highly speaker-discriminative regions can potentially be more effective than a large perturbation applied uniformly.

---

# 3. Explicitly Separate Source, Filter, and Residual Perturbations

The model should have three clearly separated branches:

```text
E ----------------→ ΔE
A(z) --------------→ ΔA
residual features -→ δ_res
```

Then:

```text
E' = P_E(E)
A' = P_A(A)
x_structured = SourceFilter(E', A')

x_protected = x_structured + δ_res
```

The important constraint is:

```text
||δ_res|| << structured perturbation magnitude
```

The raw waveform residual should not become the main attack.

## Mandatory ablation

Report:

1. SF-only
2. SF + tiny residual
3. unconstrained residual

The ideal result is:

```text
SF + tiny residual ≈ full model
```

If this happens, Medusa's source-filter contribution becomes much more credible.

---

# 4. Couple the Source and Vocal-Tract Perturbations

The current system largely treats source and filter perturbations independently.

Instead use:

```text
[ΔE, ΔA] = f_phi(E, A, F0, V)
```

rather than:

```text
ΔE = f(E)
ΔA = g(A)
```

The reason is that source excitation and vocal-tract response interact.

The model should be able to learn:

> For this speaker, pitch, phonetic context, and formant configuration, perturb the source rather than shifting the formant.

This makes Medusa an integrated source-filter attack rather than a collection of unrelated attack heads.

---

# 5. Replace the Sinusoidal SQ Phase Warp with a Better Glottal Model

The current implementation uses:

```text
phase' = phase + sq_shift * sin(freq)
```

This is useful as an experimental mechanism, but it is only an indirect approximation to an actual glottal-flow/SQ parameterization.

A stronger approach is to generate the glottal source from interpretable parameters, for example an LF-type model.

Potential parameters include:

- open quotient
- closing quotient
- asymmetry
- spectral tilt
- return phase
- glottal pulse duration.

Then:

```text
glottal parameters
        ↓
glottal waveform E
        ↓
source-filter synthesis
```

The model can perturb these parameters directly.

This makes the scientific statement much stronger:

> Medusa changes physically interpretable glottal/source parameters.

rather than:

> Medusa applies a phase warp that approximately corresponds to SQ.

---

# 6. Make LPC Perturbation Stable by Using Reflection Coefficients

Directly perturbing LPC coefficients:

```text
a'_i = a_i + Δa_i
```

can produce unstable or highly destructive all-pole filters.

Instead parameterize through reflection coefficients:

```text
kappa_i ∈ (-1, 1)
```

and map:

```text
kappa → LPC coefficients
```

using Levinson-Durbin / lattice-style parameterization.

Then learn:

```text
Δkappa_i
```

rather than unconstrained `Δa_i`.

Use:

```text
kappa'_i = tanh(raw_kappa_i)
```

to guarantee the coefficient is bounded.

This is a technically important upgrade because your experiments already showed that large LPC offsets can produce severe degradation.

---

# 7. Add a Multi-Encoder Speaker Objective

The current evaluation/training is too dependent on ECAPA.

The new objective should use an ensemble:

```text
ECAPA
WavLM-based speaker encoder
ResNet speaker encoder
CAM++
ERes2Net
```

Conceptually:

```math
L_speaker = Σ_i w_i L_speaker^(i)
```

The perturbation is therefore not optimized for one embedding space.

## Important training split

Use:

```text
training encoders:
    several known encoders

held-out encoders:
    never exposed during optimization
```

The held-out encoders measure real transferability.

---

# 8. Optimize Through the Cloning Models, Not Only Through Speaker Encoders

This is one of the most important conceptual changes.

Current style:

```text
protected audio
      ↓
speaker encoder
      ↓
embedding loss
```

Better:

```text
protected audio
      ↓
cloning model
      ↓
generated clone
      ↓
speaker encoder
      ↓
speaker-identity loss
```

Then optimize:

```math
L_clone =
SpeakerSim(x, G(x_protected))
```

rather than only:

```math
SpeakerSim(x, x_protected)
```

This directly optimizes the failure mode you actually care about:

> the attacker generates a bad speaker clone.

Use multiple differentiable or differentiably approximated surrogate cloning models during training.

Then hold out other cloning models entirely.

---

# 9. Use Multiple Cloning Models During Training

Represent the attacker as an ensemble:

```text
G = {G1, G2, G3, ...}
```

and optimize:

```math
L_attack =
E_G[L_clone(G(M_theta(x)))]
```

Recommended design:

### Surrogate models

Use several architectures during training.

### Held-out models

Use different architectures only for evaluation.

The key result becomes:

```text
strong on training attackers
+
strong on unseen attackers
```

This is much stronger than an XTTS-only result.

---

# 10. Train Through Purification and Adaptive Transformations

This is arguably the most important robustness upgrade.

The current threat model is effectively:

```text
protected audio → cloning model
```

A stronger attacker is:

```text
protected audio
      ↓
purifier / denoiser / codec / filtering
      ↓
cloning model
```

Therefore train against:

```math
x' = M_theta(x)
```

then:

```math
\tilde{x} = P_k(x')
```

and optimize:

```math
L_attack(G_k(P_k(M_theta(x))))
```

where `P_k` is randomly sampled from a family of transformations.

## Transformation family

Include:

- denoising
- low-pass filtering
- high-pass filtering
- spectral smoothing
- resampling
- gain changes
- clipping
- compression
- codec reconstruction
- speech enhancement
- mild reverberation
- background noise.

The principle is:

> the perturbation should survive transformations that a realistic attacker can apply.

---

# 11. Use an EOT-Style Training Formulation

Instead of optimizing for one fixed path:

```text
M → G
```

optimize the expected attack performance over a distribution:

```math
P ~ P_transform
G ~ P_clone
S ~ P_encoder
```

and minimize:

```math
E_{P,G,S}
[
L_speaker(G(P(M_theta(x))))
]
```

This gives Medusa robustness to:

- unknown cloning architectures
- unknown speaker encoders
- unknown preprocessing
- purification
- compression.

---

# 12. Add Explicit Content Preservation

Speaker protection should not happen because Medusa destroys phonetic content.

Introduce a frozen ASR/content representation:

```text
c(x)
c(x_protected)
```

and define:

```math
L_content = D(c(x), c(x_protected))
```

Possible content constraints:

- ASR embedding distance
- transcript consistency
- WER
- phoneme posterior distance
- speech-content encoder similarity.

The attack should move speaker information while leaving linguistic information stable.

The ideal geometry is:

```text
speaker representation:     far
content representation:     close
perceptual representation:  close
```

---

# 13. Use a Constrained Optimization Formulation

Your previous experiments showed that increasing attack strength can simply destroy audio quality.

Instead of tuning many arbitrary loss weights, formulate the problem as:

```math
min_theta L_speaker
```

subject to:

```math
L_content ≤ ε_c
L_perceptual ≤ ε_p
||δ|| ≤ ε_delta
```

You can solve this with:

- augmented Lagrangian optimization
- primal-dual optimization
- projected optimization.

This is cleaner than repeatedly searching loss weights.

---

# 14. Replace Hard Clamps with Smooth Bounded Parameterizations

Your original experiment showed β saturating at the clamp.

Instead of:

```text
β ∈ [-0.35, 0.35]
```

use:

```math
β = β_max tanh(u)
```

where `u` is unconstrained.

Apply the same idea to:

- spectral tilt
- SQ parameters
- LPC perturbations
- EQ gains
- residual amplitude.

This prevents optimization behavior from being dominated by arbitrary hard boundaries.

---

# 15. Add F0-Aware Perturbation

Introduce explicit pitch perturbation:

```math
F0' = F0 + ΔF0
```

but constrain it tightly.

Use:

```math
L_F0 = D(F0(x), F0(x'))
```

so the optimizer cannot simply change pitch drastically.

The useful idea is not:

> change pitch enough to destroy identity.

It is:

> make small changes to speaker-relevant excitation statistics while preserving natural prosody.

---

# 16. Add Harmonic-Aware Perturbation

Instead of the current fixed eight-band EQ, exploit harmonic structure.

For harmonic `k`:

```math
f_k = k F0
```

learn perturbations around harmonic amplitudes/phases:

```math
ΔH_k
Δφ_k
```

This is more naturally connected to the glottal source than fixed frequency bands.

Potential representation:

```text
F0
 ↓
harmonic frequencies
 ↓
harmonic amplitudes/phases
 ↓
structured perturbation
```

---

# 17. Add Temporal Localization

Learn a perturbation mask:

```math
m_t ∈ [0,1]
```

and:

```math
x'_t = x_t + m_t δ_t
```

with a sparsity/complexity penalty.

The optimizer can learn to perturb only:

- highly speaker-discriminative frames
- voiced segments
- specific vowel regions
- high-value harmonics.

This could significantly improve perceptual efficiency.

---

# 18. Add Explicit Psychoacoustic Masking

Your current perceptual losses are useful, but a stronger approach is to make masking explicit.

Define a frequency/time-dependent masking threshold:

```math
T_mask(f,t)
```

and constrain:

```math
|Δ(f,t)| ≤ α T_mask(f,t)
```

The model is therefore encouraged to place perturbation under the natural masking threshold of the speech itself.

This gives a more direct definition of imperceptibility than simple MSE/SNR.

---

# 19. Introduce a Small Residual Network Instead of a Free Noise Vector

If the waveform residual is retained, make it controlled.

Instead of:

```text
learn arbitrary delta
```

use:

```text
speech representation
       ↓
small residual decoder
       ↓
δ_res
```

with:

```math
δ_res = γ tanh(r_ψ(z))
```

and a very small `γ`.

The structured source-filter branch should carry almost all of the attack.

---

# 20. Universal + Speaker-Adaptive Perturbation

A strong future design is:

```math
δ = δ_universal + δ_speaker + δ_frame
```

### Universal

Works for everyone.

### Speaker-adaptive

Uses a short enrollment/reference sample.

### Frame-adaptive

Changes across the utterance.

This gives three levels of personalization:

```text
global robustness
       +
speaker specificity
       +
local speech specificity
```

It may outperform purely universal or purely per-sample optimization.

---

# 21. Channel and Codec Robustness

The protected waveform may be:

- compressed
- resampled
- streamed
- normalized
- clipped
- re-recorded.

Therefore incorporate transformations into training and benchmark evaluation.

Example:

```text
Medusa
 ↓
Opus/AAC/MP3
 ↓
16 kHz resample
 ↓
gain normalization
 ↓
cloning model
```

Protection should survive reasonable channel transformations.

---

# 22. Improve the Differentiable Source-Filter Front End

The LPC analysis is a useful starting point, but Medusa's main scientific claim would benefit from a better analysis/synthesis pipeline.

Consider:

- windowing
- overlap-add
- pre-emphasis/de-emphasis
- explicit voicing detection
- pitch-synchronous processing
- frame-consistent source/filter parameters
- stable all-pole synthesis.

The goal is to avoid treating vanilla LPC as a perfect physiological decomposition.

A more defensible formulation is:

```text
speech
 →
glottal source representation
 +
vocal-tract representation
 +
pitch/voicing
 →
structured synthesis
```

---

# 23. Make Speaker Identity and Content Explicitly Separate

The target geometry should be:

```text
                 speaker information
                        ↑
                        │
protected audio ────────┼──────── original
                        │
                        ↓
                 content information
```

More formally:

```math
D_s(s(x),s(x')) → large
```

while:

```math
D_c(c(x),c(x')) → small
```

and:

```math
D_p(x,x') → small
```

This is the central optimization principle of Medusa v2.

---

# 24. Recommended Medusa v2 Architecture

```text
                         INPUT WAVEFORM x
                                │
                                ▼
                Differentiable Source-Filter Analysis
                                │
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼
             E                 A(z)              F0 / V
             │                  │                  │
             └──────────────────┼──────────────────┘
                                ▼
                    Speech Context Encoder
                                │
                                ▼
                  Content-Conditioned Controller
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
    Δ Glottal Params       Δ Vocal-Tract        Δ Harmonic/F0
          │                     │                     │
          └─────────────────────┼─────────────────────┘
                                ▼
                    Structured Source-Filter
                           Synthesis
                                │
                                ▼
                    Small Residual Network
                                │
                                ▼
                       Protected Audio x'
```

Training/evaluation loop:

```text
                         x'
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ASV ensemble      ASR/content      perceptual model
        │                │                │
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
               random transforms / purifier
                         │
                         ▼
               surrogate cloning models
                         │
                         ▼
                  cloned speech
                         │
                         ▼
                  ASV evaluation
```

---

# 25. Recommended Loss

A reasonable high-level objective is:

```math
L =
λ_s L_speaker
+
λ_g L_clone
+
λ_c L_content
+
λ_p L_psycho
+
λ_q L_quality
+
λ_r L_robust
+
λ_θ L_parameter
```

where:

### Speaker objective

Make speaker identity difficult to recover.

### Clone objective

Make generated clones fail to preserve the original speaker.

### Content objective

Keep linguistic content unchanged.

### Psychoacoustic objective

Keep perturbation masked/inconspicuous.

### Quality objective

Preserve naturalness and waveform/spectral fidelity.

### Robust objective

Survive purification, codecs, filtering and other transformations.

### Parameter regularization

Prevent physically implausible source/filter changes.

---

# 26. The Most Important Ablation Study

Do not only compare different loss weights.

Run a structural ablation:

| Model | Source | Filter | F0/Harmonic | Residual | Expected role |
|---|---|---|---|---|---|
| A | ✗ | ✗ | ✗ | ✓ | Generic waveform baseline |
| B | ✓ | ✗ | ✗ | ✓ | Source contribution |
| C | ✗ | ✓ | ✗ | ✓ | Filter contribution |
| D | ✓ | ✓ | ✗ | ✓ | Source-filter |
| E | ✓ | ✓ | ✓ | ✓ | Full structured model |
| F | ✓ | ✓ | ✓ | tiny | Recommended final |

The critical result is:

```text
E/F substantially better than A
```

at equal perceptual cost.

That establishes that source-filter structure is actually useful.

---

# 27. Second Critical Ablation: Transferability

Train:

```text
1 ASV + 1 TTS
```

and test:

```text
multiple unseen ASV/TTS systems
```

Then compare against:

```text
ensemble training
```

This demonstrates whether the multi-model objective actually fixes the ECAPA/XTTS mismatch identified in the previous experiments.

---

# 28. Third Critical Ablation: Purification

Compare:

```text
No purification
Denoising
Spectral filtering
Codec
Strong purification
Adaptive purification
```

The important result is not only low speaker similarity before purification.

It is:

```text
low speaker similarity AFTER purification
```

---

# 29. Fourth Critical Ablation: Residual Dependence

Plot:

```text
Residual energy
        vs
Protection effectiveness
```

The desired curve is:

```text
strong protection
with very small residual energy
```

This prevents reviewers from saying:

> The method is just adversarial waveform noise plus a source-filter implementation.

---

# 30. Fifth Critical Ablation: Global vs Frame-Conditioned

Compare:

```text
36 global parameters
frame-conditioned parameters
frame + speaker-conditioned parameters
```

Measure:

- protection
- perceptual distortion
- computational cost
- transferability.

This directly tests whether the adaptive architecture actually provides value.

---

# 31. What NOT to Do

Avoid turning Medusa into:

```text
36 → 100 → 500 → 1000 arbitrary parameters
```

That is unlikely to give you a strong scientific story.

Also avoid making the main claim:

> Medusa obtains the lowest cosine similarity.

That is too dependent on the evaluator.

Instead the claim should be:

> **Medusa achieves transferable cloning disruption while preserving content and perceptual quality, using a structured source-filter perturbation that remains effective under realistic transformations and purification.**

---

# 32. Recommended Development Order

## Phase 1 — Fix the training objective

Implement:

```text
multi-ASV objective
+
multi-TTS surrogate objective
+
content preservation
```

Keep the existing 36-parameter model.

Purpose: determine whether the new objective alone improves transferability.

---

## Phase 2 — Add purification robustness

Implement random differentiable/approximate transformations:

```text
denoise
filter
resample
codec
gain
noise
```

Train through them.

Purpose: move from a brittle perturbation to robust protection.

---

## Phase 3 — Replace global parameters

Implement:

```text
θ_t = f_phi(E_t, A_t, F0_t, V_t, context_t)
```

Purpose: make Medusa content/frame adaptive.

---

## Phase 4 — Replace approximate SQ perturbation

Introduce a real glottal-flow parameterization.

Purpose: strengthen the physiological interpretation of the model.

---

## Phase 5 — Stabilize LPC perturbation

Switch from direct LPC coefficient offsets to reflection-coefficient perturbation.

Purpose: prevent destructive filter behavior.

---

## Phase 6 — Add harmonic/F0 branch

Introduce harmonic-aware source perturbations.

Purpose: improve speaker-specific attack efficiency.

---

## Phase 7 — Add psychoacoustic masking

Use frequency/time dependent masking constraints.

Purpose: improve the protection-quality frontier.

---

## Phase 8 — Constrain the residual

Keep only a very small residual branch.

Purpose: ensure the structured source-filter mechanism is the main contribution.

---

# 33. The New Research Hypothesis

The paper should ultimately test this hypothesis:

> **Speaker identity can be disrupted more efficiently by perturbing a differentiable, physiologically structured source-filter representation than by applying unconstrained waveform/frequency perturbations.**

The evidence should be:

```text
same perceptual quality
          ↓
stronger black-box protection

and

same protection
          ↓
lower perceptual distortion
```

The second comparison is particularly valuable.

---

# 34. The Ideal Final Medusa System

```text
                         ┌──────────────────┐
                         │     Speech x     │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │ Differentiable analysis │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┼────────────────┐
                 ▼               ▼                ▼
                 E              A(z)             F0/V
                 │               │                │
                 └───────────────┼────────────────┘
                                 ▼
                     Content/context encoder
                                 │
                                 ▼
                 ┌────────────────────────────────┐
                 │ Adaptive perturbation controller│
                 └───────────────┬────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        Glottal attack      Filter attack      Harmonic/F0
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
                    Source-filter synthesis
                                 │
                                 ▼
                    Tiny psychoacoustic residual
                                 │
                                 ▼
                           Protected x'
                                 │
       ┌─────────────────────────┼────────────────────────┐
       ▼                         ▼                        ▼
  content preserved       perceptually hidden      attacker robust
       │                         │                        │
       └─────────────────────────┼────────────────────────┘
                                 ▼
                 cloning failure across unseen models
```

---

# 35. Bottom Line

The biggest improvement is **not adding more attack parameters**.

The correct evolution is:

```text
Current Medusa
36 global structured parameters
        +
raw waveform delta
        ↓
Medusa v2
content-conditioned source-filter perturbation
        +
multi-ASV objective
        +
multi-TTS objective
        +
purification/EOT robustness
        +
stable physiological parameterization
        +
psychoacoustic constraint
        +
tiny residual
```

The three changes to implement first are:

```text
1. Multi-ASV + multi-TTS training
2. Purification-aware EOT training
3. Frame/content-conditioned source-filter perturbation
```

These attack the biggest weaknesses in the current model without throwing away the architecture you have already built.
