"""Pre-registered confirmatory run (report/PREREGISTRATION.md): filtered vs random, seeds 100-129.
For run_retry.sh; finished runs are skipped on restart. Prints validation metrics only."""
import sys

from generator_comparison import augment_experiment

sys.argv = ["augment_experiment", "generator_comparison/final_pool/pool.npz",
            "generator_comparison/final_pool/scores.npz", "--seeds", "30", "--seed-start", "100",
            "--ratio", "1.0", "--conditions", "filtered,random", "--tag", "confirmatory"]
augment_experiment.main()
