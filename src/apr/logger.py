from pathlib import Path
import csv, time, yaml

FIELDS = ["episode", "reward", "epsilon"]

def _run_dir(tag: str) -> Path:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("outputs", "runs", f"{ts}_{tag}")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    return run_dir

class RunLogger:
    """
    * writes cfg to YAML so runs are reproducible
    * streams episode metrics to CSV
    * exposes .dir so train loop can stash checkpoints
    """

    def __init__(self, cfg: dict, tag: str):
        self.dir = _run_dir(tag)
        (self.dir / "config.yaml").write_text(yaml.safe_dump(cfg))
        self._csv_fh = open(self.dir / "metrics.csv", "w", newline="")
        self._writer = csv.DictWriter(self._csv_fh, fieldnames=FIELDS)
        self._writer.writeheader()

    def log(self, episode: int, reward: float, epsilon: float):
        self._writer.writerow(
            {"episode": episode, "reward": reward, "epsilon": epsilon}
        )
        self._csv_fh.flush()