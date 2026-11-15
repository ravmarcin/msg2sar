"""Build a lightweight relation table between SAR bursts and MSG image pairs."""

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


_IMAGE_TIME = re.compile(r"(\d{8})T?(\d{6})(?:\.(\d+))?")


def parse_acquisition_time(value):
    """Extract a timestamp from MSG/SAR names or parse common timestamp strings."""
    match = _IMAGE_TIME.search(str(value))
    if match:
        timestamp = datetime.strptime("".join(match.group(1, 2)), "%Y%m%d%H%M%S")
        fractional = match.group(3)
        if fractional:
            timestamp = timestamp.replace(microsecond=int((fractional + "000000")[:6]))
        return timestamp
    for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            pass
    raise ValueError(f"Cannot parse acquisition timestamp from {value!r}")


class MsgSarRelationDataset:
    """Relate each configured SAR burst to a bracketing, preprocessed MSG pair.

    Records store paths and timestamps rather than raster values. The ``stages``
    field is an extensible mapping for results produced by later MSG/SAR steps.
    """

    def __init__(self, msg_pairs, msg_input_base, datasets, sar_config_path):
        self.msg_pairs = msg_pairs.copy()
        self.msg_input_base = Path(msg_input_base)
        self.datasets = list(datasets)
        self.sar_config_path = Path(sar_config_path)
        with self.sar_config_path.open("r", encoding="utf-8") as stream:
            self.sar_config = json.load(stream)
        self.sar_settings = self.sar_config.get("data", {}).get("sar", {})
        self.bursts = self.sar_settings.get("bursts", [])
        ref_date = self.sar_settings.get("ref_date")
        self.reference_time = parse_acquisition_time(ref_date) if ref_date else None
        self._pair_records = self._prepare_msg_pairs()
        self.data = pd.DataFrame(self._build_records())

    def _prepare_msg_pairs(self):
        pairs = []
        for row_index, row in self.msg_pairs.iterrows():
            before_name = row["before_name"]
            after_name = row["after_name"]
            before_time = parse_acquisition_time(before_name)
            after_time = parse_acquisition_time(after_name)
            if after_time <= before_time:
                continue
            folder = (
                row_index.strftime("%Y%m%d_%H%M%S")
                if hasattr(row_index, "strftime") else str(row_index)
            )
            pairs.append({
                "row_index": folder,
                "before_name": before_name,
                "after_name": after_name,
                "before_time": before_time,
                "after_time": after_time,
            })
        return pairs

    def _select_pair(self, target_time):
        candidates = [
            pair for pair in self._pair_records
            if pair["before_time"] <= target_time <= pair["after_time"]
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda pair: pair["after_time"] - pair["before_time"])

    def _resolve_sar_path(self, burst_name):
        download_dir = self.sar_settings.get("download_folder")
        data_child_dir = self.sar_settings.get("data_child_dir")
        if not download_dir or not data_child_dir:
            return None
        # DATA_DIR is the repository's configured data root; resolve without
        # importing project setup so this relation builder stays lightweight.
        repo_data = Path(__file__).resolve().parents[3] / "data"
        root = repo_data / data_child_dir / download_dir
        if not root.is_dir():
            return None
        matches = sorted(path for path in root.rglob("*") if path.is_file() and burst_name in path.name)
        return str(matches[0]) if matches else None

    def _build_records(self):
        records = []
        for burst_name in self.bursts:
            sar_time = parse_acquisition_time(burst_name)
            pair = self._select_pair(sar_time)
            is_reference = bool(
                self.reference_time and sar_time.date() == self.reference_time.date()
            )
            msg_images = {}
            if pair:
                duration = (pair["after_time"] - pair["before_time"]).total_seconds()
                fraction = (sar_time - pair["before_time"]).total_seconds() / duration
                for dataset in self.datasets:
                    pair_dir = self.msg_input_base / dataset / pair["row_index"]
                    before_path = pair_dir / f'{pair["before_name"]}_{dataset}.tif'
                    after_path = pair_dir / f'{pair["after_name"]}_{dataset}.tif'
                    msg_images[dataset] = {
                        "before_name": pair["before_name"],
                        "before_time": pair["before_time"].isoformat(),
                        "before_path": str(before_path),
                        "after_name": pair["after_name"],
                        "after_time": pair["after_time"].isoformat(),
                        "after_path": str(after_path),
                        "target_time": sar_time.isoformat(),
                        "target_fraction": fraction,
                        "row_index": pair["row_index"],
                        "before_exists": before_path.is_file(),
                        "after_exists": after_path.is_file(),
                    }
            records.append({
                "relation_id": burst_name,
                "sar_burst_name": burst_name,
                "sar_burst_path": self._resolve_sar_path(burst_name),
                "sar_time": sar_time.isoformat(),
                "is_sar_reference": is_reference,
                "sar_reference_time": self.reference_time.isoformat() if self.reference_time else None,
                "msg_pair_found": pair is not None,
                "msg_images": msg_images,
                "stages": {},
            })
        return records

    def to_dataframe(self):
        """Return the relation records as a DataFrame."""
        return self.data.copy()

    def record_stage(self, relation_id, stage_name, result, dataset=None):
        """Attach a processing result to a relation record for downstream stages."""
        matches = self.data.index[self.data["relation_id"] == relation_id]
        if len(matches) == 0:
            raise KeyError(f"Unknown relation: {relation_id}")
        index = matches[0]
        records = self.data.at[index, "stages"] or {}
        if dataset is not None:
            records.setdefault(stage_name, {}).setdefault(dataset, []).append(result)
        else:
            records.setdefault(stage_name, []).append(result)
        self.data.at[index, "stages"] = records

    def save(self, path):
        """Save relation records as JSON Lines, preserving nested dataset maps."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_json(path, orient="records", lines=True, date_format="iso", indent=None)
        return path
