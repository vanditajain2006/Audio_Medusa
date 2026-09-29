import os
import torch

print("--- 🕵️ Comprehensive Model Diagnostic ---\n")

models_to_test = {
    "1. Voice Conversion Model": '/content/Speech-Backbones/DiffVC/checkpts/vc/vc_libritts_wodyn.pt',
    "2. Vocoder": '/content/Speech-Backbones/DiffVC/checkpts/vocoder/generator',
    "3. Speaker Encoder": '/content/Speech-Backbones/DiffVC/checkpts/spk_encoder/pretrained.pt'
}

for name, path in models_to_test.items():
    print(f"Testing {name}...")
    if not os.path.exists(path):
        print(f"❌ FAILED: File missing entirely.")
        continue

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"   Size: {size_mb:.2f} MB")

    try:
        # The ultimate test: can PyTorch actually open it?
        _ = torch.load(path, map_location='cpu', weights_only=False)
        print(f"✅ PASSED: PyTorch successfully loaded the file.\n")
    except Exception as e:
        print(f"❌ FAILED: Corrupted file! This is the one causing the crash.")
        print(f"   Error: {e}\n")
