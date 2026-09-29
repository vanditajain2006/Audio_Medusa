Project Title: Audio Medusa - The Invisible Defence Against Voice Cloning

The Problem: "They Can Steal Your Voice"

We live in an era where AI can replicate a person's voice with terrifying accuracy. With just 3 seconds of audio recorded from a phone call, a YouTube video, or a social media post, malicious AI models can "clone" your voice.
Once cloned, hackers can use your voice to break into bank accounts, scam your family members, or create deepfakes that make you say things you never said. The current solution is to detect these fakes, but that is a losing battle. We need a way to stop the cloning from happening in the first place.

# Audio Medusa  
### Multi-Component Voice Protection Against Audio Deepfakes

> **Audio Medusa** is an audio deepfake-protection project that modifies important voice characteristics to make a person’s speech harder to clone, imitate, or synthesize using AI voice-generation systems.

Modern voice-cloning models do not learn a speaker’s identity from only one property such as pitch or accent. They learn a combination of harmonic patterns, vocal-fold behavior, spectral characteristics, timing variations, phase information, and other subtle acoustic cues.

Audio Medusa addresses this problem through a **multi-component protection approach**. Instead of adding simple noise to an audio file, it makes controlled changes to several voice-related features that are important for representing speaker identity. The goal is to reduce the effectiveness of deepfake systems while keeping the speech understandable and reasonably natural for human listeners.

---

## What Does Audio Medusa Do?

Audio Medusa takes an input speech recording and applies targeted transformations to selected acoustic components of the voice.

```text
Original Voice Recording
          │
          ▼
Audio Feature Analysis
          │
          ▼
Multi-Component Voice Protection
          │
          ▼
Protected Voice Recording
```

The protected output should ideally:

- Preserve the spoken message and overall intelligibility.
- Retain acceptable speech quality for human listeners.
- Change speaker-specific acoustic patterns.
- Make it more difficult for a voice-cloning model to learn or reproduce the original speaker’s identity.
- Reduce the similarity between a deepfake-generated voice and the original speaker.

---

## How Does It Work?

Human speech is created through multiple interacting processes:

1. The vocal folds generate a source signal through vibration.
2. The vocal tract, including the throat, mouth, tongue, and lips, filters this source signal.
3. Pitch, amplitude, harmonics, noise, timing, and phase evolve over time to create a unique voice.

Deepfake and voice-cloning systems learn these patterns from recorded speech. Audio Medusa analyzes selected speech features and applies controlled perturbations to them. Rather than heavily distorting the entire recording, it focuses on features that contribute to speaker identity.

```text
Input Speech
    │
    ▼
Feature Extraction
    ├── Harmonic and aperiodic information
    ├── Pitch-period variation
    ├── Amplitude variation
    ├── Source–filter characteristics
    ├── Phase information
    └── Glottal excitation properties
    │
    ▼
Controlled Feature Perturbation
    │
    ▼
Audio Reconstruction
    │
    ▼
Protected Speech
```

The overall approach aims for a balance between:

```text
Higher deepfake resistance
        +
Lower speaker-identity leakage
        +
Preserved speech intelligibility
        +
Acceptable audio quality
```

---

## Protection Components

Audio Medusa focuses on five important components of human speech.

### 1. HAR: Harmonic–Aperiodic Relations

Speech contains both:

- **Harmonic components**, which are periodic and strongly related to vocal-fold vibration.
- **Aperiodic components**, which are noise-like and arise from breathiness, turbulence, unvoiced sounds, and natural voice variation.

HAR-based protection modifies the relationship between harmonic and aperiodic information in the audio signal. Since these patterns can contribute to speaker identity, changing them can make the voice more difficult for deepfake models to reproduce accurately.

**Purpose:** Reduce the stability of harmonic and noise-related cues used to represent a speaker’s voice.

---

### 2. Jitter and Shimmer

Jitter and shimmer describe very small natural variations in human speech.

- **Jitter** is the cycle-to-cycle variation in pitch period or vocal-fold vibration timing.
- **Shimmer** is the cycle-to-cycle variation in speech amplitude or loudness.

These tiny variations contribute to the individuality and texture of a person’s voice. Audio Medusa applies controlled modifications to jitter and shimmer characteristics so that deepfake models cannot easily learn a perfectly stable representation of the speaker.

**Purpose:** Alter micro-level pitch and amplitude patterns that may reveal speaker identity.

---

### 3. Source–Filter Characteristics

Human speech is commonly described using the source–filter model:

```text
Speech Signal = Glottal Source × Vocal-Tract Filter
```

- The **source** is the excitation created by vocal-fold vibration.
- The **filter** is the shaping effect of the vocal tract, including formants and resonances.

The source–filter component of Audio Medusa modifies selected excitation and filtering cues. This can affect speaker-specific spectral patterns without fully destroying the spoken content.

**Purpose:** Make speaker-specific excitation, formant, and resonance characteristics harder to model.

---

### 4. Phase Relations

An audio signal contains both magnitude and phase information.

- **Magnitude** tells us how much energy exists at different frequencies.
- **Phase** describes the timing relationship between frequency components.

Many audio-processing and voice-cloning methods focus strongly on magnitude information. However, phase can also contain useful structural cues about the waveform. Audio Medusa explores controlled phase modifications to alter these relationships while maintaining usable audio quality.

**Purpose:** Disrupt waveform-level timing and frequency relationships that may support voice reconstruction.

---

### 5. Glottal Symmetry

Glottal symmetry is related to the shape and balance of vocal-fold movement during speech production. It influences the glottal excitation signal and can affect voice qualities such as breathiness, tension, spectral tilt, and vocal texture.

Since glottal behavior differs across speakers, it can carry important identity-related information. Audio Medusa modifies glottal-symmetry-related properties to make this information less consistent and less useful for voice-cloning models.

**Purpose:** Obscure speaker-specific vocal-fold and excitation characteristics.

---

## Project Objective

The main objective of Audio Medusa is to create a voice-protection mechanism that changes speaker-identifying characteristics while preserving the semantic content of speech.

In simple terms:

> A human listener should still understand the message, but an AI voice-cloning system should have more difficulty reproducing the original speaker.

Audio Medusa is designed as a research-oriented framework for studying how different acoustic components contribute to voice cloning and how controlled transformations can help defend against audio deepfakes.

---

## Ethical Use

Audio Medusa is intended for responsible research, voice privacy, deepfake defense, and educational experimentation.

It should be used only with appropriate consent and should not be used for impersonation, unauthorized voice manipulation, or harmful activities.     
