"""Run Experiments 1-2 only (Feature Comparison + Classifier Comparison)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
from run_academic_experiments import (
    load_all_datasets, run_feature_comparison, run_classifier_comparison,
)

def main():
    t_start = time.time()
    print("=" * 72)
    print("  EXPERIMENTS 1-2: Feature + Classifier Comparison")
    print("=" * 72)
    datasets = load_all_datasets()
    if not datasets:
        print("ERROR: No datasets found.")
        return
    exp1 = run_feature_comparison(datasets)
    exp2 = run_classifier_comparison(datasets)
    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min ({elapsed:.0f}s)")

if __name__ == "__main__":
    main()
