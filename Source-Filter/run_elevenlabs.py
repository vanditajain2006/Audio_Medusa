import os
import torch
import torchaudio
import torchaudio.transforms as T
from speechbrain.inference.speaker import SpeakerRecognition
from medusa_v2_pipeline import train_one, EXPERIMENTS, SR, DEVICE

def main():
    audio_file = "ElevenLabs_Confident_female_voice_proclaiming__We_will_succeed_together!.wav"
    if not os.path.exists(audio_file):
        print(f"File not found: {audio_file}")
        return
        
    print(f"Applying Medusa Adaptive SF Protection to: {audio_file}")
    
    print("Loading ASV Ensemble...")
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
    
    # Pre-build shared transforms
    mfcc_t = T.MFCC(sample_rate=SR, n_mfcc=40).to(DEVICE)
    mel_ts = [
        T.MelSpectrogram(sample_rate=SR, n_fft=n, hop_length=h, n_mels=80).to(DEVICE)
        for n, h in [(512, 128), (1024, 256), (2048, 512)]
    ]
    
    # Load and resample
    waveform, sr = torchaudio.load(audio_file)
    if sr != SR:
        waveform = T.Resample(sr, SR)(waveform)
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    waveform = waveform.to(DEVICE)
    
    with torch.no_grad():
        embs_orig = {v_name: v.encode_batch(waveform) for v_name, v in verifiers.items()}
        
    config = EXPERIMENTS["HYB_strong"]
    out_file = "protected_elevenlabs.wav"
    
    print(f"Starting optimization for {config.get('iterations', 800)} iterations...")
    protected = train_one(
        waveform, config, verifiers, embs_orig, mfcc_t, mel_ts
    )
    
    # Save the output
    import soundfile as sf
    sf.write(out_file, protected.squeeze().cpu().numpy(), SR)
    print(f"\nProtection complete! Saved to {out_file}")

if __name__ == "__main__":
    main()
