# Audio Medusa: Hybrid Source-Filter Defense

**Your voice is your identity. This project makes it harder for AI to steal.**

Modern zero-shot voice-conversion models (like XTTS, YourTTS, and ElevenLabs) can take a few seconds of someone's speech and re-render it in any other person's voice, or turn anyone's speech into a convincing imitation of a target speaker. That power is being abused for fraud calls, impersonation scams, fake audio evidence and non-consensual deepfakes. 

Audio Medusa's **Hybrid Source-Filter Defense** is a research toolkit that fights back at the source: it adds an **imperceptible protective layer to a recording** so that when an AI model attempts to extract your voice identity, the result breaks down.

---

## Why it matters

- **Voice cloning is now cheap, accessible, and zero-shot.** Open-source models need only a short 3-second reference clip to steal an identity.
- **Detection is reactive; protection is proactive.** Deepfake detectors work after the damage is done. Protecting audio *before* it is published or shared stops the misuse at the point of creation.
- **It puts control back with the speaker.** Podcasters, creators, journalists, and public speakers can shield their recordings without changing how they sound to listeners.
- **It attacks the fundamental math of speech generation.** Studying how generative speech models fail under adversarial input helps build more robust systems and better defenses.

## How it works

Voice-conversion models separate a voice into two parts: *content* (what is being said) and *identity/timbre* (who is saying it). This project exploits that dependency in three steps using a **Hybrid CNN-based Adaptive Source-Filter Model**:

1. **Adaptive Source-Filter Perturbation.** Instead of just adding random noise, we use a 1D-CNN that learns to manipulate the fundamental frequencies (the "source") and the vocal tract resonances (the "filter"). This scrambles the speaker's biometric footprint while maintaining the spoken words.
2. **Hybrid Waveform Attack.** We combine the structured Source-Filter manipulation with a constrained `delta` waveform perturbation. This allows the defense to attack both the harmonic structure and the raw spectral envelope simultaneously.
3. **Whisper Content Preservation.** We use the encoder from OpenAI's Whisper model to actively monitor the *linguistic content* during optimization. The system is mathematically forced to ensure that the protected audio still perfectly matches the original transcript, preventing the audio from degrading into static.

To a human listener, the protected clip sounds like the original. To the voice-conversion model, the biometric identity features are scrambled, so the cloned output sounds degraded, metallic, or completely loses the target's likeness.

## What is inside

| File | Purpose |
|---|---|
| `medusa_v2_pipeline.py` | **Main Engine**: Contains the CNN AdaptiveSourceFilterModule, the hybrid training loop, Whisper content loss, and the full evaluation pipeline. |
| `evaluate_5_audios.py` | Runs a batch 3-way evaluation across different configurations (`HYB_strong`, `CTRL_sf_moderate`, `HYB_optimal`). |
| `run_elevenlabs.py` | A dedicated script for testing the defense specifically against ElevenLabs API cloning. |
| `download_5_audios.py` | Utility to fetch standardized test audio samples. |
| `sf_final_exp.ipynb` | Jupyter Notebook containing the experimental setup, graph visualizations, and early prototyping. |
| `till_now.md` & `walkthrough.md` | Developer notes, architecture decisions, and codebase evolution history. |
| `Medusa_Model_Improvement_Recommendations.md` | Comprehensive audit report of the system's strengths and mathematical bottlenecks. |

## Quick start

1. **Install Requirements:** Make sure you have PyTorch, torchaudio, and speechbrain installed, as well as a local installation of TTS (for YourTTS/XTTS cloning).
2. **Download Samples:** Run `python download_5_audios.py` to get the baseline dataset.
3. **Run the Defense Pipeline:** 
   ```bash
   python evaluate_5_audios.py
   ```
   This will train the protection model, apply the defense, and automatically evaluate the results using ECAPA-TDNN speaker verification and Mel-Cepstral Distortion (MCD) metrics.

## Tunable parameters (in `medusa_v2_pipeline.py`)

| Parameter | Effect |
|---|---|
| `model_type` | Choose between `hybrid` (SF + delta), `sf_only` (pure source-filter), or `delta_only`. |
| `max_perturb_ratio` | The maximum amplitude of the waveform attack. Higher = stronger defense, but more audible distortion. |
| `content_preserve` | Weight of the Whisper-based content loss. Higher = cleaner speech, lower = stronger identity disruption. |
| `mel_preserve` | Weight of the spectrogram fidelity constraint to keep the audio sounding natural. |
| `augmentation` | Enables Expectation Over Transformation (EOT) to make the defense robust against MP3 compression and resampling. |

## Honest scope

This is a research prototype. The current `HYB_strong` configuration achieves a **~26% reduction in clone similarity** and a **100% Equal Error Rate (EER)** against modern cloners, meaning it successfully destroys the target identity. However, this comes at the cost of perceptual quality (PESQ ~1.05). Hiding the adversarial noise perfectly from human ears while maintaining clone disruption remains an open challenge in adversarial audio research. Use it only on audio you own or have permission to protect.

## Acknowledgements

Built utilizing insights from adversarial machine learning and the [SpeechBrain](https://speechbrain.github.io/) toolkit for speaker verification scoring.
