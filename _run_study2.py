"""Study 2 (report/PREREGISTRATION_2.md): 4 pools x 10 seeds x 4 conditions = 160 runs, fixed step budget,
deterministic kernels. pool_k uses seeds 200 + 10 (k - 1) ... 209 + 10 (k - 1); results in
generator_comparison/experiment/study2/pool<k>/. For run_retry.sh; finished runs are skipped on restart.
Prints validation metrics only.

  --val-only   same runs without the test split, into experiment/study2_check/ (Phase 3 determinism check);
               pass e.g. --val-only 1 200 filtered to run just pool 1, seed 200, filtered."""
import sys

from generator_comparison import augment_experiment

POOLS = (1, 2, 3, 4)
SEEDS_PER_POOL = 10
FIRST_SEED = 200
CONDITIONS = "real,oversample,random,filtered"
STEPS, EVAL_EVERY = 6250, 250
POOL_DIR = "generator_comparison/pools2"


def args_for(k, seed_start, n_seeds, conditions, tag_root, val_only):
    return [f"{POOL_DIR}/pool_{k}.npz", f"{POOL_DIR}/scores_{k}.npz", "--seed-start", str(seed_start),
            "--seeds", str(n_seeds), "--ratio", "1.0", "--conditions", conditions, "--steps", str(STEPS),
            "--eval-every", str(EVAL_EVERY), "--deterministic", "--tag", f"{tag_root}/pool{k}"] \
        + (["--val-only"] if val_only else [])


if __name__ == "__main__":
    if sys.argv[1:2] == ["--val-only"]:
        k, seed, cond = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        augment_experiment.main(args_for(k, seed, 1, cond, "study2_check", val_only=True))
    else:
        for k in POOLS:
            augment_experiment.main(args_for(k, FIRST_SEED + SEEDS_PER_POOL * (k - 1), SEEDS_PER_POOL, CONDITIONS,
                                             "study2", val_only=False))
