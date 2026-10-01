import os
import medusa_v2_pipeline

def main():
    # Override AUDIO_FILES to point to the 5 speakers
    medusa_v2_pipeline.AUDIO_FILES = [f"audio_samples_5/ls_{i}.wav" for i in range(1, 6)]

    # Run all three: HYB_optimal trains fresh, others skip (cached files exist)
    medusa_v2_pipeline.EXPERIMENTS = {
        "CTRL_sf_moderate": medusa_v2_pipeline.EXPERIMENTS["CTRL_sf_moderate"],
        "HYB_strong":       medusa_v2_pipeline.EXPERIMENTS["HYB_strong"],
        "HYB_optimal":      medusa_v2_pipeline.EXPERIMENTS["HYB_optimal"],
    }

    print("\nStarting 3-way comparison: CTRL_sf_moderate | HYB_strong | HYB_optimal")
    df = medusa_v2_pipeline.run_batch_pipeline_v5()

    print("\n\n" + "="*80)
    print("FINAL 3-WAY COMPARISON:")
    print("="*80)
    print(df.to_string())

if __name__ == "__main__":
    main()
