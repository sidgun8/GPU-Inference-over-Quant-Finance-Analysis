from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


class LocalLogger:
    def __init__(self, results_dir: str | Path, run_id: str) -> None:
        self.base_results_dir = Path(results_dir)
        self.run_id = run_id
        self.results_dir = self.base_results_dir / "runs" / run_id
        self.latest_dir = self.base_results_dir / "latest"
        self.raw_dir = self.results_dir / "raw"
        self.tables_dir = self.results_dir / "tables"
        self.figures_dir = self.results_dir / "figures"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []

    def log_record(self, record: dict[str, Any]) -> Path:
        self.records.append(record)
        record["run_id"] = self.run_id
        name = f"{record['run_id']}_{record['model']}_{record['backend']}_{record['precision']}_bs{record['batch_size']}_rep{record['repeat_index']}.json"
        path = self.raw_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
        return path

    def write_summary(self) -> Path:
        path = self.tables_dir / "benchmark_summary.csv"
        pd.DataFrame(self.records).to_csv(path, index=False)
        self._write_latest_copy()
        return path

    def _write_latest_copy(self) -> None:
        if self.latest_dir.exists():
            shutil.rmtree(self.latest_dir)
        shutil.copytree(self.results_dir, self.latest_dir)
