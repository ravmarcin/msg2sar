from .stack_base import MsgStackBase
from pathlib import Path
from tqdm import tqdm
import numpy as np
import traceback
from datetime import datetime
import pyresample as pr
from pyresample.bilinear import NumpyBilinearResampler
from libs.internal.img.optical_temporal_upsampling import estimate_tvl1_flow, warp_with_flow
import rasterio
from pathlib import Path
import numpy as np

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pyresample.kd_tree')


class MsgTemporalResample(MsgStackBase):

    def __init__(
            self,
            data_dir: str,
            work_dir: str,
            geojson_path: str = None,
            projection: dict = None,
            resolution: float = 0.005,
            spatial_buffer_meters: int = 40000
        ) -> None:
        super().__init__(
            data_dir=data_dir,
            work_dir=work_dir,
            geojson_path=geojson_path,
            projection=projection,
            resolution=resolution,
            spatial_buffer_meters=spatial_buffer_meters
        )

    def estimate_flow_for_pair(
            self,
            dataset: str,
            row=None,
            row_index: str = None,
            before_name: str = None,
            after_name: str = None,
            input_base: str = None,
            nd_val: float = -9999.0,
            save_output: bool = False,
    ):
        """
        Estimate dense optical flow for two consecutive MSG images of a single dataset.

        This is the step-2 helper requested for the temporal upsampling workflow:
        it reads the pair of GeoTIFFs already produced by nat-to-tiff conversion,
        then computes a dense flow field in both directions:
        before -> after and after -> before.

        Parameters
        ----------
        dataset : str
            Dataset/channel name (for example "IR_108").
        row : pandas.Series, optional
            DataFrame row from self.data_df; used to infer before_name/after_name
            and the average-date folder name.
        row_index : str, optional
            Explicit row-folder name, e.g. "20230102_044443".
        before_name, after_name : str, optional
            Explicit file stems used to build the TIFF filenames.
        input_base : str or Path, optional
            Base directory containing nat_to_tiff outputs. Defaults to
            self.work_dir / "nat_to_tiff".
        nd_val : float
            No-data value to pass into the optical-flow estimation.

        Returns
        -------
        dict
            Dictionary with dataset, row_index, file names, and the flow arrays
            for both directions.
        """
        if input_base is None:
            input_base = Path(self.work_dir) / 'MsgStackConvert.nat_to_tif_batch'
        else:
            input_base = Path(input_base)

        if row is not None:
            row_index = row_index or row.name
            before_name = before_name or row.get('before_name')
            after_name = after_name or row.get('after_name')

        if row_index is None and row is None:
            raise ValueError('Either row or row_index must be provided.')

        if hasattr(row_index, 'strftime'):
            row_index = row_index.strftime('%Y%m%d_%H%M%S')

        if before_name is None or after_name is None:
            raise ValueError('before_name and after_name are required to locate the two TIFF inputs.')

        row_dir = input_base / dataset / str(row_index)
        before_path = row_dir / f'{before_name}_{dataset}.tif'
        after_path = row_dir / f'{after_name}_{dataset}.tif'

        if not before_path.exists():
            raise FileNotFoundError(f'Missing before image for {dataset}: {before_path}')
        if not after_path.exists():
            raise FileNotFoundError(f'Missing after image for {dataset}: {after_path}')

        with rasterio.open(str(before_path)) as src_before:
            before_arr = src_before.read(1).astype(np.float32)

        with rasterio.open(str(after_path)) as src_after:
            after_arr = src_after.read(1).astype(np.float32)

        flow_a_to_b = estimate_tvl1_flow(before_arr, after_arr, nd_val=nd_val)
        flow_b_to_a = estimate_tvl1_flow(after_arr, before_arr, nd_val=nd_val)

        saved_paths = {}
        if save_output:
            out_dir = self.process_path(dataset=dataset, row_index=str(row_index))
            out_dir.mkdir(parents=True, exist_ok=True)

            a_to_b_path = out_dir / f'{before_name}_{dataset}_a_to_b_flow.tif'
            b_to_a_path = out_dir / f'{after_name}_{dataset}_b_to_a_flow.tif'

            self._write_multiband_geotiff(str(a_to_b_path), flow_a_to_b, self.area_def, epsg=4326, no_data=nd_val)
            self._write_multiband_geotiff(str(b_to_a_path), flow_b_to_a, self.area_def, epsg=4326, no_data=nd_val)

            saved_paths = {
                'a_to_b_path': str(a_to_b_path),
                'b_to_a_path': str(b_to_a_path),
            }

        result = {
            'dataset': dataset,
            'row_index': str(row_index),
            'before_name': before_name,
            'after_name': after_name,
            'before_path': str(before_path),
            'after_path': str(after_path),
            'a_to_b': flow_a_to_b,
            'b_to_a': flow_b_to_a,
        }
        if saved_paths:
            result['saved_paths'] = saved_paths
        return result

    def estimate_flow_to_time_for_pair(
            self,
            dataset: str,
            row=None,
            row_index: str = None,
            before_name: str = None,
            after_name: str = None,
            input_base: str = None,
            nd_val: float = -9999.0,
            target_time=None,
            target_fraction: float = None,
            save_output: bool = False,
    ):
        """
        Estimate the motion field from each image toward a target timestamp between
        two consecutive images.

        This is the derived midpoint/average-time step from the raw pairwise motion:
        the raw flow is first computed as before -> after and after -> before, then it
        is linearly scaled to the target time. If no target is given, the midpoint time
        between the two images is used by default.

        Parameters
        ----------
        dataset : str
            Dataset/channel name.
        row : pandas.Series, optional
            Row containing before_datetime and after_datetime fields.
        row_index : str, optional
            Explicit row identifier.
        before_name, after_name : str, optional
            Explicit TIFF file stems.
        input_base : str or Path, optional
            Base directory containing nat_to_tiff outputs.
        target_time : datetime-like, optional
            The timestamp to which the flow should be scaled.
        target_fraction : float, optional
            Fraction of the interval from before to after. If omitted, 0.5 is used.
        save_output : bool
            If True, save the target-time flow fields to process output.
        """
        if target_time is not None and target_fraction is not None:
            raise ValueError('Specify either target_time or target_fraction, not both.')

        pair_result = self.estimate_flow_for_pair(
            dataset=dataset,
            row=row,
            row_index=row_index,
            before_name=before_name,
            after_name=after_name,
            input_base=input_base,
            nd_val=nd_val,
            save_output=False,
        )

        if row is not None:
            before_dt = row.get('before_datetime')
            after_dt = row.get('after_datetime')
        else:
            before_dt = None
            after_dt = None

        if before_dt is None or after_dt is None:
            raise ValueError('row must contain before_datetime and after_datetime to compute target-time flow.')

        if target_time is None:
            if target_fraction is None:
                target_fraction = 0.5
            if not 0.0 <= float(target_fraction) <= 1.0:
                raise ValueError('target_fraction must be between 0 and 1.')
            target_time = before_dt + (after_dt - before_dt) * float(target_fraction)
        elif isinstance(target_time, str):
            target_time = datetime.fromisoformat(target_time.replace('Z', '+00:00'))

        total_seconds = (after_dt - before_dt).total_seconds()
        if total_seconds == 0:
            alpha = 0.0
        else:
            alpha = (target_time - before_dt).total_seconds() / total_seconds

        alpha = float(np.clip(alpha, 0.0, 1.0))
        if target_fraction is None:
            target_fraction = alpha

        flow_a_to_target = pair_result['a_to_b'] * alpha
        flow_b_to_target = pair_result['b_to_a'] * (1.0 - alpha)

        saved_paths = {}
        if save_output:
            out_dir = self.process_path(dataset=dataset, row_index=str(pair_result['row_index']))
            out_dir.mkdir(parents=True, exist_ok=True)

            a_to_target_path = out_dir / f'{pair_result["before_name"]}_{dataset}_a_to_target_flow.tif'
            b_to_target_path = out_dir / f'{pair_result["after_name"]}_{dataset}_b_to_target_flow.tif'

            self._write_multiband_geotiff(str(a_to_target_path), flow_a_to_target, self.area_def, epsg=4326, no_data=nd_val)
            self._write_multiband_geotiff(str(b_to_target_path), flow_b_to_target, self.area_def, epsg=4326, no_data=nd_val)

            saved_paths = {
                'a_to_target_path': str(a_to_target_path),
                'b_to_target_path': str(b_to_target_path),
            }

        result = {
            'dataset': dataset,
            'row_index': str(pair_result['row_index']),
            'before_name': pair_result['before_name'],
            'after_name': pair_result['after_name'],
            'before_path': pair_result['before_path'],
            'after_path': pair_result['after_path'],
            'before_datetime': before_dt,
            'after_datetime': after_dt,
            'target_time': target_time,
            'target_fraction': float(target_fraction),
            'alpha': float(alpha),
            'a_to_b': pair_result['a_to_b'],
            'b_to_a': pair_result['b_to_a'],
            'a_to_target': flow_a_to_target,
            'b_to_target': flow_b_to_target,
        }
        if saved_paths:
            result['saved_paths'] = saved_paths
        return result

    def temporal_resample_for_pair(
            self,
            dataset: str,
            flow_result=None,
            row=None,
            row_index: str = None,
            before_name: str = None,
            after_name: str = None,
            input_base: str = None,
            nd_val: float = -9999.0,
            target_time=None,
            target_fraction: float = 0.5,
            save_output: bool = False,
    ):
        """
        Warp both reference images to the same target timestamp using a target-time
        flow result that has already been computed for the pair.

        The method expects the output of estimate_flow_to_time_for_pair as the
        flow_result input. If flow_result is not supplied, a convenience fallback
        computes it internally using the pair metadata and the target time.

        The outputs are written with a reference-specific suffix so that both midpoint
        estimates coexist, e.g. "..._from_before_ref.tif" and "..._from_after_ref.tif".
        """
        if flow_result is None:
            flow_result = self.estimate_flow_to_time_for_pair(
                dataset=dataset,
                row=row,
                row_index=row_index,
                before_name=before_name,
                after_name=after_name,
                input_base=input_base,
                nd_val=nd_val,
                target_time=target_time,
                target_fraction=target_fraction,
                save_output=False,
            )

        before_path = Path(flow_result['before_path'])
        after_path = Path(flow_result['after_path'])

        with rasterio.open(str(before_path)) as src_before:
            before_arr = src_before.read(1).astype(np.float32)

        with rasterio.open(str(after_path)) as src_after:
            after_arr = src_after.read(1).astype(np.float32)

        a_to_target = flow_result['a_to_target']
        b_to_target = flow_result['b_to_target']

        # The raw motion is stored as before -> after. At the midpoint, the after
        # image must be warped backwards by half the interval, which is equivalent
        # to using the same forward flow with alpha = alpha - 1.0. For a 0.5 target
        # fraction, this becomes alpha = -0.5.
        resampled_from_before = warp_with_flow(before_arr, a_to_target, alpha=1.0)
        resampled_from_after = warp_with_flow(after_arr, flow_result['a_to_b'], alpha=flow_result['alpha'] - 1.0)

        # Keep the explicit inverse-direction flow in the result for downstream use,
        # but do not re-apply it with alpha=1.0 because that would keep the image on
        # the wrong side of the interval.

        saved_paths = {}
        if save_output:
            out_dir = self.process_path(dataset=dataset, row_index=str(flow_result['row_index']))
            out_dir.mkdir(parents=True, exist_ok=True)

            before_ref_path = out_dir / f'{flow_result["before_name"]}_{dataset}_midpoint_from_before_ref.tif'
            after_ref_path = out_dir / f'{flow_result["after_name"]}_{dataset}_midpoint_from_after_ref.tif'

            self._write_geotiff(str(before_ref_path), resampled_from_before, self.area_def, epsg=4326, no_data=nd_val)
            self._write_geotiff(str(after_ref_path), resampled_from_after, self.area_def, epsg=4326, no_data=nd_val)

            saved_paths = {
                'from_before_ref_path': str(before_ref_path),
                'from_after_ref_path': str(after_ref_path),
            }

        result = {
            'dataset': dataset,
            'row_index': str(flow_result['row_index']),
            'before_name': flow_result['before_name'],
            'after_name': flow_result['after_name'],
            'before_path': str(before_path),
            'after_path': str(after_path),
            'target_time': flow_result['target_time'],
            'target_fraction': flow_result['target_fraction'],
            'alpha': flow_result['alpha'],
            'a_to_target': a_to_target,
            'b_to_target': b_to_target,
            'from_before_ref': resampled_from_before,
            'from_after_ref': resampled_from_after,
        }
        if saved_paths:
            result['saved_paths'] = saved_paths
        return result

    def estimate_flow_for_date(self, dataset: str, date_value, input_base: str = None, nd_val: float = -9999.0):
        """Estimate flow for every consecutive pair in a single date bucket."""
        if input_base is None:
            input_base = Path(self.work_dir) / 'MsgStackConvert.nat_to_tif_batch'
        else:
            input_base = Path(input_base)

        if isinstance(date_value, str):
            if len(date_value) == 8:
                target_day = datetime.strptime(date_value, '%Y%m%d').date()
            else:
                target_day = datetime.fromisoformat(date_value).date()
        else:
            target_day = date_value.date()

        pair_results = []
        for idx, row in self.data_df.iterrows():
            try:
                row_day = idx.date() if hasattr(idx, 'date') else datetime.strptime(str(idx), '%Y%m%d_%H%M%S').date()
            except Exception:
                continue

            if row_day != target_day:
                continue

            pair_results.append(self.estimate_flow_for_pair(
                dataset=dataset,
                row=row,
                input_base=input_base,
                nd_val=nd_val,
            ))

        return pair_results

    def temporal_upsample_batch(self, datasets: list, input_base: str = None,
                               output_base: str = None, nd_val: float = -9999.0,
                               overwrite: bool = False, sar_config_path: str = None,
                               clamp_alpha: bool = True):
        """
        Perform temporal upsampling / flow estimation for multiple channels (datasets).

        Parameters
        ----------
        datasets : list
            List of dataset/channel names to process (e.g., ['IR_108','VIS006']).
        input_base : str, optional
            Base input directory containing nat_to_tiff outputs. Defaults to
            self.work_dir / 'nat_to_tiff'.
        output_base : str, optional
            Base output directory to write flows and differences. Defaults to
            self.work_dir / 'temporal_upsampling'.
        nd_val : float
            No-data value used in images.
        overwrite : bool
            Overwrite existing outputs if True.

        Returns
        -------
        dict
            Summary of processing: total, processed, skipped, failed list.
        """
        if input_base is None:
            input_base = Path(self.work_dir) / 'MsgStackConvert.nat_to_tif_batch'
        else:
            input_base = Path(input_base)

        if output_base is None:
            output_base = self.process_path()
        else:
            output_base = Path(output_base)

        results = {'total_rows': 0, 'processed': 0, 'skipped': 0, 'failed': []}

        # Load SAR bursts if provided
        bursts = []
        if sar_config_path is not None:
            try:
                import json
                with open(sar_config_path, 'r') as sf:
                    cfg = json.load(sf)
                    bursts = cfg.get('data', {}).get('sar', {}).get('bursts', [])
            except Exception:
                bursts = []

        for idx, row in tqdm(self.data_df.iterrows(), total=len(self.data_df), desc='Temporal upsampling'):
            results['total_rows'] += 1
            row_index_str = idx.strftime('%Y%m%d_%H%M%S') if hasattr(idx, 'strftime') else str(idx)
            # expect per-dataset row dirs under input_base/<dataset>/<row_index>
            available_ds = []
            missing_datasets = []

            try:
                # Load before/after images for all datasets
                before_arrays = {}
                after_arrays = {}
                for ds in datasets:
                    ds_row_dir = input_base / ds / row_index_str
                    if not ds_row_dir.exists():
                        missing_datasets.append(ds)
                        continue

                    bf_path = ds_row_dir / f"{row['before_name']}_{ds}.tif"
                    af_path = ds_row_dir / f"{row['after_name']}_{ds}.tif"
                    if not bf_path.exists() or not af_path.exists():
                        missing_datasets.append(ds)
                        continue

                    try:
                        with rasterio.open(str(bf_path)) as src:
                            before_arrays[ds] = src.read(1).astype(np.float32)
                        with rasterio.open(str(af_path)) as src:
                            after_arrays[ds] = src.read(1).astype(np.float32)
                        available_ds.append(ds)
                    except Exception as e:
                        missing_datasets.append(ds)
                        continue

                # If no datasets available for this row, record failure and continue
                if len(available_ds) == 0:
                    results['failed'].append({'row_index': row_index_str, 'error': 'no datasets available', 'missing_datasets': missing_datasets})
                    continue

                # Ensure shapes match
                shapes = {k: v.shape for k, v in before_arrays.items()}
                if len(set(shapes.values())) != 1:
                    raise ValueError('Input channel shapes do not match')

                # Estimate flow per channel
                flows = {}
                for ds in datasets:
                    flow = estimate_tvl1_flow(before_arrays[ds], after_arrays[ds], nd_val=nd_val)
                    flows[ds] = flow  # shape (H, W, 2)

                # Compute differences: per-channel vs mean of others
                diffs = {}
                for ds in datasets:
                    others = [flows[o] for o in datasets if o != ds]
                    if others:
                        mean_other = np.mean(np.stack(others, axis=0), axis=0)
                        diff_vec = flows[ds] - mean_other
                        diff_mag = np.sqrt(diff_vec[..., 0] ** 2 + diff_vec[..., 1] ** 2)
                    else:
                        diff_mag = np.zeros(flows[ds].shape[:2], dtype=np.float32)
                    diffs[ds] = diff_mag

                # Save flows and diffs per dataset under output_base/dataset/row_index/
                for ds in datasets:
                    out_dir = output_base / ds / row_index_str
                    out_dir.mkdir(parents=True, exist_ok=True)

                    before_stem = f"{row['before_name']}_{ds}"
                    after_stem = f"{row['after_name']}_{ds}"

                    flow_before_path = out_dir / f"{before_stem}_flow.tif"
                    diff_before_path = out_dir / f"{before_stem}_diff.tif"

                    flow_after_path = out_dir / f"{after_stem}_flow.tif"
                    diff_after_path = out_dir / f"{after_stem}_diff.tif"

                    # If outputs exist and not overwrite, skip
                    if (flow_before_path.exists() or flow_after_path.exists()) and not overwrite:
                        results['skipped'] += 1
                        continue

                    # Use area_def from base to write
                    # flows: (H, W, 2)
                    self._write_multiband_geotiff(str(flow_before_path), flows[ds], self.area_def, epsg=4326, no_data=nd_val)
                    self._write_geotiff(str(diff_before_path), diffs[ds], self.area_def, epsg=4326, no_data=nd_val)

                    # For after we can write the same flow (flow from before->after), but mark file with after_stem
                    self._write_multiband_geotiff(str(flow_after_path), flows[ds], self.area_def, epsg=4326, no_data=nd_val)
                    self._write_geotiff(str(diff_after_path), diffs[ds], self.area_def, epsg=4326, no_data=nd_val)

                # For each SAR burst, compute alpha and create temporally upsampled MSG from before and after
                if bursts:
                    for burst in bursts:
                        # Parse burst timestamp like YYYYMMDDTHHMMSS inside the burst string
                        import re
                        m = re.search(r'(\d{8}T\d{6})', burst)
                        if not m:
                            # try YYYYMMDD_HHMMSS or YYYYMMDDHHMMSS
                            m2 = re.search(r'(\d{8}_\d{6}|\d{14})', burst)
                            if m2:
                                ts_str = m2.group(1).replace('_', 'T')
                            else:
                                continue
                        else:
                            ts_str = m.group(1)

                        try:
                            from datetime import datetime
                            burst_dt = datetime.strptime(ts_str, '%Y%m%dT%H%M%S')
                        except Exception:
                            continue

                        before_dt = row.get('before_datetime')
                        after_dt = row.get('after_datetime')
                        if before_dt is None or after_dt is None:
                            continue

                        total_sec = (after_dt - before_dt).total_seconds()
                        if total_sec == 0:
                            alpha = 0.0
                        else:
                            alpha = (burst_dt - before_dt).total_seconds() / total_sec
                        if clamp_alpha:
                            alpha = float(max(0.0, min(1.0, alpha)))

                        burst_id = ts_str.replace('T', '')

                        for ds in datasets:
                            # only process if dataset was available
                            if ds not in flows:
                                continue

                            bf_arr = before_arrays.get(ds)
                            af_arr = after_arrays.get(ds)
                            flow = flows[ds]

                            # warp from before forward
                            up_from_before = warp_with_flow(bf_arr, flow, alpha)
                            # warp from after backward (alpha - 1)
                            up_from_after = warp_with_flow(af_arr, flow, alpha - 1.0)

                            ds_out_dir = output_base / ds / row_index_str
                            ds_out_dir.mkdir(parents=True, exist_ok=True)

                            up_before_path = ds_out_dir / f"{row['before_name']}_{ds}_upsampled_from_before_{burst_id}.tif"
                            up_after_path = ds_out_dir / f"{row['after_name']}_{ds}_upsampled_from_after_{burst_id}.tif"

                            self._write_geotiff(str(up_before_path), up_from_before, self.area_def, epsg=4326, no_data=nd_val)
                            self._write_geotiff(str(up_after_path), up_from_after, self.area_def, epsg=4326, no_data=nd_val)

                results['processed'] += 1
            except Exception as e:
                results['failed'].append({'row_index': row_index_str, 'error': f"{str(e)}\n{traceback.format_exc()}"})

        return results
        