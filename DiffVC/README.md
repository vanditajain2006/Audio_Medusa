# DiffVC Shield: Protecting Your Voice from AI Cloning

**Your voice is your identity. This project makes it harder for AI to steal.**

Modern voice-conversion models can take a few seconds of someone's speech and re-render it in any other
person's voice, or turn anyone's speech into a convincing imitation of a target speaker. That power is being
abused for fraud calls, impersonation scams, fake audio evidence and non-consensual deepfakes. DiffVC Shield is a
research toolkit that fights back at the source: it adds an **imperceptible protective layer to a recording**
so that when a diffusion-based voice-conversion model tries to process it, the result breaks down.

---

## Why it matters

- **Voice cloning is now cheap and accessible.** Open-source models like
  [DiffVC](https://github.com/huawei-noah/Speech-Backbones/tree/main/DiffVC) need only a short reference clip.
- **Detection is reactive; protection is proactive.** Deepfake detectors work after the damage is done.
  Protecting audio *before* it is published or shared stops the misuse at the point of creation.
- **It puts control back with the speaker.** Podcasters, creators, journalists, public speakers and anyone
  posting voice notes or videos can shield their recordings without changing how they sound to listeners.
- **It is a testbed for AI-safety research.** Studying how generative speech models fail under adversarial
  input helps build more robust systems and better defenses.

## How it works

Voice-conversion models do not listen to raw audio. They first turn it into a **mel-spectrogram**, a
picture of which frequencies are active over time, and everything downstream depends on that picture.
This project exploits that dependency in three steps:

1. **Mirror the model's front-end.** We rebuild DiffVC's audio-to-mel pipeline in PyTorch so it is
   *differentiable*, meaning we can compute how a tiny change in the waveform changes the spectrogram the model sees.
2. **Optimize a protective perturbation.** In the *Spectrogram Shredder*, we search for a very small waveform change
   whose spectrogram is pushed toward a **frequency-flipped** version of the original (low and high frequencies
   swapped). Using sign-gradient updates, the change is hard-capped at a maximum amplitude (`epsilon`) so it stays
   near-inaudible to people.
3. **Hide it where the ear won't notice.** A **loudness-adaptive perceptual mask** places more of the perturbation
   in louder passages and less in quiet ones, keeping the protected file natural-sounding.

To a listener the protected clip sounds like the original. To the voice-conversion model, its input features are
scrambled, so the cloned output is degraded instead of a clean imitation.

The project also includes a second, alternative approach, a **mel-space attack with padding/cropping**, which
perturbs the spectrogram directly and steers the DiffVC generator's output away from its clean result.
It is included for comparison and research.

## What is inside

| File | Purpose |
|---|---|
| `01_clone_and_patch_diffvc.txt` | Clones the DiffVC source and patches it to work with current library versions |
| `02_check_checkpoints.py` | Verifies that the three pretrained model files (voice conversion, vocoder, speaker encoder) load correctly |
| `03_setup_models_baseline.py` | Loads all models and generates a normal, unprotected voice clone as a baseline |
| `04_mel_attack_padding.py` | Alternative protection: mel-space attack with padding/cropping |
| `05_spectrogram_shredder.py` | **Main method**: waveform-level Spectrogram Shredder, then side-by-side clone comparison |
| `06_save_outputs.py` | Saves the protected source, normal clone and blocked clone as WAV files |

## Quick start

Run this in Google Colab (GPU runtime recommended):

1. Run the shell commands in `01_clone_and_patch_diffvc.txt`. Restart the session afterwards.
2. Run `02_check_checkpoints.py` to confirm the models load.
3. Run `05_spectrogram_shredder.py` (it includes model setup, so it works on its own).
   You will hear four clips: original, protected source, normal clone, and blocked clone.
4. Run `06_save_outputs.py` to export the audio files.

`04_mel_attack_padding.py` needs the functions from `03` or `05` to be loaded first.

The scripts use Colab paths (`/content/Speech-Backbones/DiffVC`), so adjust them if you run elsewhere.

## Tunable parameters

| Parameter | Effect |
|---|---|
| `epsilon` | Maximum perturbation strength. Higher gives stronger protection but more audible change |
| `lr` | Step size of each optimization update |
| `iterations` | Number of optimization steps. More gives a more refined perturbation and takes longer |

## Honest scope

This is a research prototype, not a guarantee. The protection is tuned against one model (DiffVC, LibriTTS
checkpoint), and heavy compression, re-recording, resampling or noise removal, as well as other
voice-conversion systems, may weaken it. Treat it as a foundation for building and testing stronger voice-protection
techniques, and use it only on audio you own or have permission to protect.

## Acknowledgements

Built on [DiffVC](https://github.com/huawei-noah/Speech-Backbones/tree/main/DiffVC) by Huawei Noah's Ark Lab and
the HiFi-GAN vocoder. Upstream code and checkpoints are not included in this repository. Please review their licenses.
