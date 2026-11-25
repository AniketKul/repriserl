"""Run a bounded, single-device arithmetic GRPO experiment."""
import argparse
from pathlib import Path

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=["baseline", "fixed-replay"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory for a new run. Existing results are preserved.")
    parser.add_argument("--tiny", action="store_true", help="Train an offline random model on CPU.")
    parser.add_argument("--sizes", type=int, nargs=3, metavar=("TRAIN", "DEV", "TEST"),
                        help="Number of disjoint arithmetic problems in each split.")
    parser.add_argument("--eval-every", type=int, default=1, metavar="STEPS")
    parser.add_argument("--num-iterations", type=int, choices=(1, 2, 4), default=1,
                        help="Number of stock GRPO updates per generation batch.")
    args = parser.parse_args()
    sizes = tuple(args.sizes or ((32, 2, 4) if args.tiny else (20_000, 512, 2_048)))
    config = load_config(args.config)
    if args.arm == "fixed-replay" and args.num_iterations != 1:
        parser.error("fixed replay requires --num-iterations 1")
    if args.arm == "baseline":
        from .adapters.trl.baseline import run_baseline
        result = run_baseline(config, args.output, tiny=args.tiny, sizes=sizes,
                              eval_every=args.eval_every, num_iterations=args.num_iterations)
    else:
        from .adapters.trl.replay import run_fixed_replay
        result = run_fixed_replay(config, args.output, tiny=args.tiny, sizes=sizes,
                                  eval_every=args.eval_every)
    print(f"{result['arm']} {result['status']} after {result['steps']} updates. "
          f"Results in {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
