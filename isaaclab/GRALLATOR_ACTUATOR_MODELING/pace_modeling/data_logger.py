"""Bounded asynchronous CSV and metadata writer."""

from __future__ import annotations

import csv
import json
import queue
import threading
from pathlib import Path


class PaceDataLogger:
    def __init__(self, csv_path, fieldnames, metadata, queue_size=4096):
        self.csv_path = Path(csv_path)
        self.metadata_path = self.csv_path.with_name(
            self.csv_path.stem + "_metadata.json"
        )
        self.fieldnames = list(fieldnames)
        self.queue = queue.Queue(maxsize=int(queue_size))
        self.error = None
        self.rows_written = 0
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.thread = threading.Thread(target=self._writer, name="PaceCsvWriter")
        self.thread.start()

    def _writer(self):
        try:
            with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.fieldnames)
                writer.writeheader()
                while True:
                    row = self.queue.get()
                    if row is None:
                        break
                    writer.writerow(row)
                    self.rows_written += 1
        except Exception as exc:  # surfaced synchronously by write/close
            self.error = exc

    def write(self, row):
        if self.error is not None:
            raise RuntimeError(f"PACE CSV writer failed: {self.error}")
        try:
            self.queue.put_nowait(dict(row))
        except queue.Full as exc:
            raise RuntimeError("PACE CSV queue is full; refusing to lose samples") from exc

    def close(self):
        self.queue.put(None)
        self.thread.join()
        if self.error is not None:
            raise RuntimeError(f"PACE CSV writer failed: {self.error}")

