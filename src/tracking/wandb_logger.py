from __future__ import annotations

from pathlib import Path
from typing import Any


class WandbLogger:
    def __init__(self, config: dict[str, Any], group: str) -> None:
        self.config = config
        self.enabled = bool(config.get("enabled", False))
        self.group = group
        self.wandb = None
        if self.enabled:
            import wandb

            self.wandb = wandb
            if config.get("require_login", True):
                if wandb.api.api_key is None:
                    raise RuntimeError("W&B is enabled with require_login=true, but no W&B API key/login is available.")

    def log_record(self, record: dict[str, Any]) -> None:
        if not self.enabled or self.wandb is None:
            return
        run = self.wandb.init(
            project=self.config["project"],
            entity=self.config.get("entity"),
            mode=self.config.get("mode", "online"),
            group=self.group,
            name=f"{record['model']}-{record['backend']}-{record['precision']}-bs{record['batch_size']}-rep{record['repeat_index']}",
            config=record,
            reinit=True,
        )
        run.log(record)
        run.finish()

    def log_artifacts(self, paths: list[Path]) -> None:
        if not self.enabled or self.wandb is None or not self.config.get("log_artifacts", True):
            return
        run = self.wandb.init(
            project=self.config["project"],
            entity=self.config.get("entity"),
            mode=self.config.get("mode", "online"),
            group=self.group,
            name=f"{self.group}-artifacts",
            reinit=True,
        )
        artifact = self.wandb.Artifact(f"{self.group}-artifacts", type="benchmark-results")
        for path in paths:
            if path.exists():
                artifact.add_file(str(path))
        run.log_artifact(artifact)
        run.finish()
