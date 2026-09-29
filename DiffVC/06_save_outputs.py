import torchaudio

# 1. The ORIGINAL SOURCE is already a file at src_path ('example/6415_111615_000012_000005.wav')

# 2. Save SECURED SOURCE
torchaudio.save('/content/secured_source.wav', poisoned_audio_tensor.cpu(), 22050)

# 3. Save NORMAL CLONE
# .unsqueeze(0) is needed because torchaudio expects shape (channels, time)
torchaudio.save('/content/normal_clone.wav', normal_clone.cpu().unsqueeze(0), 22050)

# 4. Save BLOCKED CLONE
torchaudio.save('/content/blocked_clone.wav', attacked_clone.cpu().unsqueeze(0), 22050)

print("✅ Files saved successfully for evaluation.")
