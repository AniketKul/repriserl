"""Run bounded fixed replay through the stock single-device trainer."""
from pathlib import Path

from repriserl.config import RunConfig


def run_fixed_replay(config: RunConfig, output_dir: Path, *, tiny: bool = False,
                     sizes: tuple[int, int, int] = (20_000, 512, 2048), eval_every: int = 1) -> dict:
    from .baseline import run_training
    from .compat import ReplayGRPOAdapter

    return run_training(config, output_dir, tiny=tiny, sizes=sizes, num_iterations=1,
                        trainer_class=ReplayGRPOAdapter, arm="fixed_replay", eval_every=eval_every)
