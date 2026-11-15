"""Apply the FILM temporal interpolation model to MSG image pairs."""

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

from .stack_base import MsgStackBase
from libs.internal.img.film_temporal_upsampling import (
    film_temporal_upsampling,
    load_film_model,
)


class MsgFilmTemporalResample(MsgStackBase):
    """Upsample MSG pairs and preserve their georeferencing in output TIFFs."""

    def __init__(self, *args, temporal_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_config = self._read_config(temporal_config)
        self._film_model = None

    @staticmethod
    def _read_config(config):
        if config is None:
            return {}
        if isinstance(config, dict):
            return config
        with open(config, "r", encoding="utf-8") as stream:
            return json.load(stream)

    @staticmethod
    def _timestamp_from_name(name):
        match = re.search(r"(\d{14})(?:\.(\d+))?", str(name))
        if not match:
            raise ValueError(f"No YYYYMMDDHHMMSS timestamp found in image name: {name}")
        timestamp = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
        fractional = match.group(2)
        if fractional:
            timestamp = timestamp.replace(microsecond=int((fractional + "000000")[:6]))
        return timestamp

    def _settings(self):
        settings = self.temporal_config.get("msg_processing", self.temporal_config)
        settings = settings.get("temporal_upsampling", settings)
        return settings.get("film", settings)

    def _get_model(self, settings):
        if self._film_model is None:
            model_path = settings.get("model_path")
            if not model_path:
                raise ValueError("Set msg_processing.temporal_upsampling.film.model_path in the config")
            model_path = Path(model_path).expanduser()
            if not model_path.is_absolute():
                model_path = Path(__file__).resolve().parents[4] / model_path
            self._film_model = load_film_model(model_path)
        return self._film_model

    def temporal_upsample_pair(
        self, dataset, row, input_base=None, output_base=None,
        target_fraction=None, overwrite=False, target_time=None,
    ):
        """Interpolate a pair at a temporal fraction derived from file timestamps."""
        settings = self._settings()
        input_base = Path(input_base or Path(self.work_dir) / "MsgStackConvert.nat_to_tif_batch")
        output_base = Path(output_base or self.process_path(method_name="film_temporal_upsampling"))
        before_name, after_name = row["before_name"], row["after_name"]
        before_time = self._timestamp_from_name(before_name)
        after_time = self._timestamp_from_name(after_name)
        if after_time <= before_time:
            raise ValueError(f"Image timestamps must increase: {before_name}, {after_name}")

        if target_time is not None:
            if isinstance(target_time, str):
                target_time = datetime.fromisoformat(target_time.replace("Z", "+00:00"))
            target_fraction = (target_time - before_time).total_seconds() / (after_time - before_time).total_seconds()
        elif target_fraction is None:
            raise ValueError("Provide target_time from the SAR acquisition or an explicit target_fraction")
        target_fraction = float(target_fraction)
        if not 0.0 <= target_fraction <= 1.0:
            raise ValueError("The target time must fall between the MSG image timestamps")
        target_time = before_time + (after_time - before_time) * target_fraction
        row_index = row.name.strftime("%Y%m%d_%H%M%S") if hasattr(row.name, "strftime") else str(row.name)
        input_dir = input_base / dataset / row_index
        before_path = input_dir / f"{before_name}_{dataset}.tif"
        after_path = input_dir / f"{after_name}_{dataset}.tif"
        if not before_path.is_file() or not after_path.is_file():
            raise FileNotFoundError(f"Missing input pair for {dataset}: {before_path}, {after_path}")

        output_dir = output_base / dataset / row_index
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (
            f"{before_name}_{dataset}_film_{target_time.strftime('%Y%m%dT%H%M%S_%f')}.tif"
        )
        if output_path.exists() and not overwrite:
            return {"status": "skipped", "path": str(output_path), "target_time": target_time}

        with rasterio.open(before_path) as before_src, rasterio.open(after_path) as after_src:
            if (before_src.height, before_src.width) != (after_src.height, after_src.width):
                raise ValueError("MSG pair dimensions do not match")
            before = before_src.read(1).astype(np.float32)
            after = after_src.read(1).astype(np.float32)
            profile = before_src.profile.copy()
            before_nodata, after_nodata = before_src.nodata, after_src.nodata

        vmin, vmax = self._channel_range(settings, dataset)
        result = self._interpolate(before, after, target_fraction, vmin, vmax, settings)
        nodata = before_nodata if before_nodata is not None else after_nodata
        if nodata is not None:
            before_invalid = ~np.isfinite(before) | (before == before_nodata if before_nodata is not None else False)
            after_invalid = ~np.isfinite(after) | (after == after_nodata if after_nodata is not None else False)
            result[before_invalid & after_invalid] = nodata
        profile.update(driver="GTiff", count=1, dtype="float32", compress="deflate")
        if nodata is not None:
            profile.update(nodata=nodata)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(result.astype(np.float32), 1)

        return {
            "status": "processed", "dataset": dataset, "before_name": before_name,
            "after_name": after_name, "target_time": target_time,
            "target_fraction": target_fraction, "path": str(output_path),
        }

    def _interpolate(self, before, after, fraction, vmin, vmax, settings):
        return film_temporal_upsampling(
            before, after, fraction, vmin, vmax, self._get_model(settings)
        )

    def temporal_upsample_relations(
        self, relation_dataset, input_base=None, output_base=None, overwrite=False,
    ):
        """Interpolate each relation's MSG pair at its SAR acquisition timestamp."""
        if hasattr(relation_dataset, "to_dataframe"):
            records = relation_dataset.to_dataframe().to_dict(orient="records")
        elif hasattr(relation_dataset, "to_dict"):
            records = relation_dataset.to_dict(orient="records")
        else:
            records = list(relation_dataset)
        summary = {"total": 0, "processed": 0, "skipped": 0, "failed": []}
        for relation in tqdm(records, desc="FILM interpolation at SAR times"):
            if not relation.get("msg_pair_found"):
                summary["failed"].append({
                    "relation_id": relation.get("relation_id"),
                    "error": "No MSG image pair brackets the SAR acquisition time",
                })
                continue
            for dataset, images in relation.get("msg_images", {}).items():
                summary["total"] += 1
                row_data = {
                    "before_name": images["before_name"],
                    "after_name": images["after_name"],
                }
                # The pair's source directory is supplied directly so the
                # relation builder can be used independently of MsgStackBase.
                try:
                    class PairRow:
                        name = images["row_index"]

                        def __getitem__(self, key):
                            return row_data[key]

                    result = self.temporal_upsample_pair(
                        dataset=dataset,
                        row=PairRow(),
                        input_base=input_base,
                        output_base=output_base,
                        target_time=relation["sar_time"],
                        overwrite=overwrite,
                    )
                    result["relation_id"] = relation.get("relation_id")
                    result["sar_burst_name"] = relation.get("sar_burst_name")
                    if hasattr(relation_dataset, "record_stage"):
                        relation_dataset.record_stage(
                            relation.get("relation_id"), "msg_film_interpolation", result, dataset
                        )
                    summary[result["status"]] += 1
                except Exception as exc:
                    summary["failed"].append({
                        "relation_id": relation.get("relation_id"),
                        "dataset": dataset, "error": str(exc),
                    })
        return summary

    @staticmethod
    def _channel_range(settings, dataset):
        ranges = settings.get("value_ranges", {})
        if dataset not in ranges:
            raise ValueError(f"Set a FILM value range for dataset {dataset!r} in the config")
        value_range = ranges[dataset]
        if isinstance(value_range, dict):
            return float(value_range["vmin"]), float(value_range["vmax"])
        return float(value_range[0]), float(value_range[1])

    def temporal_upsample_batch(
        self, datasets, input_base=None, output_base=None,
        target_fractions=None, overwrite=False,
    ):
        """Run explicit fraction-based interpolation (prefer SAR relations)."""
        if target_fractions is None:
            raise ValueError(
                "target_fractions must be supplied explicitly; use "
                "temporal_upsample_relations for SAR-derived target times"
            )
        summary = {"total": 0, "processed": 0, "skipped": 0, "failed": []}
        for _, row in tqdm(self.data_df.iterrows(), total=len(self.data_df), desc="FILM temporal upsampling"):
            for dataset in datasets:
                for fraction in target_fractions:
                    summary["total"] += 1
                    try:
                        item = self.temporal_upsample_pair(
                            dataset, row, input_base, output_base, fraction, overwrite
                        )
                        summary[item["status"]] += 1
                    except Exception as exc:
                        summary["failed"].append({
                            "dataset": dataset, "before_name": row.get("before_name"),
                            "after_name": row.get("after_name"), "error": str(exc),
                        })
        return summary
