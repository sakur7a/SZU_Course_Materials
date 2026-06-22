"""Run Experiments 3-4 only (Augmentation Ablation + Overfitting Analysis)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
from run_academic_experiments import (
    load_all_datasets, run_augmentation_ablation, run_overfitting_analysis,
)

def main():
    t_start = time.time()
    print("=" * 72)
    print("  EXPERIMENTS 3-4: Augmentation Ablation + Overfitting Analysis")
    print("=" * 72)
    datasets = load_all_datasets()
    if not datasets:
        print("ERROR: No datasets found.")
        return
    exp3 = run_augmentation_ablation(datasets)
    exp4 = run_overfitting_analysis(datasets)
    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min ({elapsed:.0f}s)")

if __name__ == "__main__":
    main()
