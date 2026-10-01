import os
import soundfile as sf
from datasets import load_dataset

out_dir = "audio_samples_5"
os.makedirs(out_dir, exist_ok=True)

print("Streaming LibriSpeech test-clean...")
ds = load_dataset("librispeech_asr", "clean", split="test", streaming=True)

count = 0
for item in ds:
    audio = item["audio"]
    # save
    out_path = os.path.join(out_dir, f"ls_{count+1}.wav")
    sf.write(out_path, audio["array"], audio["sampling_rate"])
    print(f"Saved {out_path}")
    count += 1
    if count >= 5:
        break
print("Done.")
