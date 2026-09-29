# ==========================================
# ALTERNATIVE SOLUTION (MODULE 2): Match Encoder Input Size
# ==========================================
# If the generator's encoder expects a fixed input size (like 409 frames),
# we need to either crop or pad the mel-spectrogram to match.

def run_medusa_attack_with_padding(src_path, tgt_path, iterations=150, epsilon=0.15, lr=0.01):
    """
    Alternative approach:
    1. Detect the expected mel length from the encoder
    2. Pad/crop mel_src to match
    3. Attack at that fixed size
    """
    print(f"Direct Mel Perturbation Attack with Padding/Cropping")

    from librosa.core import load

    # Get original melspecs
    mel_src_orig = get_official_mel(src_path)   # (1, 80, T_src)
    mel_tgt_orig = get_official_mel(tgt_path)   # (1, 80, T_tgt)

    print(f"Original shapes:")
    print(f"  mel_src: {mel_src_orig.shape}")
    print(f"  mel_tgt: {mel_tgt_orig.shape}")

    # Get speaker embedding
    embed_tgt = torch.from_numpy(
        spk_encoder.embed_utterance(
            spk_encoder.preprocess_wav(tgt_path)
        )
    ).float().unsqueeze(0).to(device)

    # Try different cropping strategies
    mel_src_len = mel_src_orig.shape[-1]
    mel_tgt_len = mel_tgt_orig.shape[-1]

    # Strategy: Use the minimum length (safest option)
    # This ensures both fit into encoder
    target_mel_len = min(mel_src_len, mel_tgt_len)

    print(f"\nUsing target length: {target_mel_len}")

    # Crop to target length
    mel_src = mel_src_orig[:, :, :target_mel_len]
    mel_tgt = mel_tgt_orig[:, :, :target_mel_len]

    print(f"Cropped shapes:")
    print(f"  mel_src: {mel_src.shape}")
    print(f"  mel_tgt: {mel_tgt.shape}")

    m_src_len = torch.LongTensor([mel_src.shape[-1]]).to(device)
    m_tgt_len = torch.LongTensor([mel_tgt.shape[-1]]).to(device)

    # Initialize perturbation
    delta_mel = torch.zeros_like(mel_src, requires_grad=True, device=device)

    # Perceptual masking
    mel_magnitude = torch.abs(mel_src)
    mel_mean = mel_magnitude.mean()
    mel_mask = (mel_magnitude / (mel_mean + 1e-6)).clamp(0.2, 1.0)
    mel_mask = mel_mask.detach()

    print(f"\n--- Starting PGD Attack ---")
    print(f"Delta shape: {delta_mel.shape}")
    print(f"Mask shape: {mel_mask.shape}")

    attack_losses = []

    for i in range(iterations):
        if delta_mel.grad is not None:
            delta_mel.grad.zero_()

        # Perturbed mel
        mel_adv = mel_src + (delta_mel * mel_mask)
        mel_adv = torch.clamp(mel_adv, min=-15.0, max=3.0)

        try:
            with torch.amp.autocast('cuda' if use_gpu else 'cpu'):
                # Both inputs are same shape now
                _, mel_out_clean = generator.forward(
                    mel_src, m_src_len, mel_tgt, m_tgt_len, embed_tgt,
                    n_timesteps=30, mode='ml'
                )

                _, mel_out_adv = generator.forward(
                    mel_adv, m_src_len, mel_tgt, m_tgt_len, embed_tgt,
                    n_timesteps=30, mode='ml'
                )

            loss = F.l1_loss(mel_out_adv, mel_out_clean)
            total_loss = 100.0 * loss
            attack_losses.append(total_loss.item())

        except RuntimeError as e:
            print(f"\n❌ Shape error at iteration {i}: {str(e)}")
            print(f"   Shapes: mel_src={mel_src.shape}, mel_adv={mel_adv.shape}")
            print(f"   mel_out_clean shape might differ")

            # Skip this iteration
            if (i + 1) % 30 == 0:
                print(f"  Iteration {i+1}/{iterations} | SKIPPED (shape mismatch)")
            continue

        # Backward
        if total_loss.requires_grad:
            total_loss.backward()

        if delta_mel.grad is None:
            continue

        grad_norm = delta_mel.grad.abs().mean().item()

        if grad_norm > 1e-10:
            with torch.no_grad():
                delta_mel.add_(lr * delta_mel.grad.sign())
                delta_mel.clamp_(-epsilon, epsilon)

        if (i + 1) % 30 == 0:
            delta_norm = (delta_mel * mel_mask).abs().max().item()
            print(f"  Iteration {i+1:3d}/{iterations} | Loss: {total_loss.item():.8f} | Grad: {grad_norm:.10f} | Delta: {delta_norm:.8f}")

    # Final mel
    mel_adv_final = mel_src + (delta_mel.detach() * mel_mask)
    mel_adv_final = torch.clamp(mel_adv_final, min=-15.0, max=3.0)

    # Reconstruct full-length poisoned audio
    # Pad back to original length
    mel_poisoned_full = torch.zeros_like(mel_src_orig)
    mel_poisoned_full[:, :, :target_mel_len] = mel_adv_final

    # If original was longer, copy the remainder from original (no perturbation)
    if mel_src_len > target_mel_len:
        mel_poisoned_full[:, :, target_mel_len:] = mel_src_orig[:, :, target_mel_len:]

    print(f"\nReconstructed mel shape: {mel_poisoned_full.shape}")

    # Vocoder
    with torch.no_grad():
        poisoned_audio = vocoder.forward(mel_poisoned_full).cpu().squeeze().clamp(-1, 1)

    perturbation_mag = (delta_mel.detach() * mel_mask).abs().max().item()

    print(f"\n✅ Attack complete!")
    print(f"   Poisoned audio shape: {poisoned_audio.shape}")
    print(f"   Max perturbation: {perturbation_mag:.8f}")
    print(f"   Loss progression: min={min(attack_losses):.8f}, max={max(attack_losses):.8f}, final={attack_losses[-1]:.8f}")

    return {
        'poisoned_audio': poisoned_audio,
        'delta_mel': delta_mel.detach(),
        'mel_adv': mel_adv_final.detach(),
        'perturbation_magnitude': perturbation_mag,
        'losses': attack_losses
    }

# ===================================================================
# EXECUTION
# ===================================================================

tgt_path = 'example/8534_216567_000015_000010.wav'
src_path = 'example/6415_111615_000012_000005.wav'

print("\n" + "="*70)
print("Running Mel Attack with Padding/Cropping")
print("="*70 + "\n")

try:
    result = run_medusa_attack_with_padding(
        src_path, tgt_path,
        iterations=150,
        epsilon=0.15,
        lr=0.01
    )

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print("\n1. ORIGINAL SOURCE")
    ipd.display(ipd.Audio(src_path))

    print("\n2. POISONED SOURCE")
    print(f"   Perturbation magnitude: {result['perturbation_magnitude']:.8f}")
    ipd.display(ipd.Audio(result['poisoned_audio'], rate=22050))

    print("\n3. TARGET VOICE")
    ipd.display(ipd.Audio(tgt_path))

    print("\n4. CLONE FROM ORIGINAL")
    original_clone = quick_convert(src_path, tgt_path)
    ipd.display(ipd.Audio(original_clone, rate=22050))

    print("\n5. CLONE FROM POISONED")
    poisoned_clone = quick_convert(result['poisoned_audio'], tgt_path)
    ipd.display(ipd.Audio(poisoned_clone, rate=22050))

    # Plot loss curve
    print("\n6. ATTACK LOSS CURVE")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(result['losses'], linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('L1 Loss')
    plt.title('Medusa Attack Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\n✅ If poisoned output (5) differs from original output (4),")
    print("   the watermark is working!")

except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
