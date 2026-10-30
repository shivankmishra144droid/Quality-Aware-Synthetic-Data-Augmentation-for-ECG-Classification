"""Phase 3 launcher (for run_retry.sh, which resumes by skipping finished runs):
4 conditions x 10 seeds on the final pool. 10 seeds so a two-sided Wilcoxon can reach p < 0.05."""
import sys

from generator_comparison import augment_experiment

sys.argv = ["augment_experiment", "generator_comparison/final_pool/pool.npz",
            "generator_comparison/final_pool/scores.npz", "--seeds", "10", "--ratio", "1.0",
            "--conditions", "real,random,filtered,filtered_nolabel,oversample", "--tag", "main"]
augment_experiment.main()
