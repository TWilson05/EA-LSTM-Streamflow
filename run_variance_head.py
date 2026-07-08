"""Job 2 entry point — train the post-hoc variance head. Twin of run_training.py.

Root runner for the MVE/Student-t variance head. Mirrors run_training.py but trains
ONE small head over the frozen, precomputed hidden states (no 10-member array).

Responsibility:
  - CLI/argparse: --exp_name, --model_type (NO --member_id / --save_hidden unless the
    per-member prototype path is taken).
  - Resolve feature config by importing EXPERIMENT_CONFIGS from run_training (DRY).
  - Wire src/mve_dataset -> src/mve_head -> src/mve_training -> src/mve_inference.
  - Read frozen artifacts via the index contract (data/output/results_MVE/); never
    import Job 1 training internals.

Scaffold only — bodies intentionally left unimplemented (owner to populate).

TODO:
  - def main(): argparse + orchestration
  - if __name__ == "__main__": main()
"""
