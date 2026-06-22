"""Run Experiments 5-7 only (Statistical Comparison + Ensemble + Cross-Dataset)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
from run_academic_experiments import (
    load_all_datasets, run_statistical_comparison,
    run_ensemble_experiment, run_cross_dataset_generalization,
)

def main():
    t_start = time.time()
    print("=" * 72)
    print("  EXPERIMENTS 5-7: Statistical + Ensemble + Cross-Dataset")
    print("=" * 72)
    datasets = load_all_datasets()
    if not datasets:
        print("ERROR: No datasets found.")
        return
    exp5 = run_statistical_comparison(datasets)
    exp6 = run_ensemble_experiment(datasets)
    exp7 = run_cross_dataset_generalization(datasets)
    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min ({elapsed:.0f}s)")

if __name__ == "__main__":
    main()
