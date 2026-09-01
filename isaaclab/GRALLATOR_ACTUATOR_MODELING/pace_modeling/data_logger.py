"""In-memory experiment logger flushed only after motor shutdown."""

from __future__ import annotations

import csv
import json
from pathlib import Path


class PaceDataLogger:
    def __init__(self, csv_path, fieldnames, metadata, max_rows=100_000):
        self.csv_path = Path(csv_path)
        self.metadata_path = self.csv_path.with_name(
            self.csv_path.stem + "_metadata.json"
        )
        self.fieldnames = list(fieldnames)
        self.metadata = dict(metadata)
        self.max_rows = int(max_rows)
        self.rows = []
        self.rows_written = 0
        self.closed = False
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row):
        if self.closed:
            raise RuntimeError("PACE logger is already closed")
        if len(self.rows) >= self.max_rows:
            raise RuntimeError("PACE in-memory log is full; refusing to lose samples")
        self.rows.append(row)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.metadata_path.write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        self.rows_written = len(self.rows)
