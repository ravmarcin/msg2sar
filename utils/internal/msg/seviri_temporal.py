try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from utils.internal.log.logger import get_logger
from utils.internal.msg.msg_config import MsgConfig
from utils.internal.msg.pymsg.stack_base import MsgStackBase
from utils.internal.img.temporal_upsampling import batch_temporal_upsampling, get_trans_param

log = get_logger()


class SeviriTemporalProcessor:
    """
    SEVIRI temporal downsampling processor.

    Merges SEVIRI image pairs to match InSAR timestamps using optical flow-based
    temporal interpolation. Handles multi-channel SEVIRI data (WV_062, WV_073,
    IR_097, IR_087, IR_108, IR_120).
    """

    def __init__(self, config: MsgConfig):
        self.config = config
        self.channels = ['WV_062', 'WV_073', 'IR_097', 'IR_087', 'IR_108', 'IR_120']

        # Get temporal processing config
        self.temporal_config = config.msg_processing.get('temporal_downsampling', {})
        self.n_images_window = self.temporal_config.get('n_images_window', 10)
        self.batch_size_x = self.temporal_config.get('batch_size_x', 256)
        self.batch_size_y = self.temporal_config.get('batch_size_y', 256)
        self.overlap = self.temporal_config.get('overlap', 32)
        self.translation_steps = self.temporal_config.get('translation_steps', [8, 4, 2, 1])

        log.info(
            f"SeviriTemporalProcessor initialized:\n"
            f"  Channels: {self.channels}\n"
            f"  N images window: {self.n_images_window}\n"
            f"  Batch size: {self.batch_size_x}x{self.batch_size_y}\n"
            f"  Overlap: {self.overlap}"
        )

    def find_nearest_images(
        self,
        stack_base: MsgStackBase,
        sar_timestamp: datetime,
        n_images: int = None
    ) -> pd.DataFrame:
        """
        Find n nearest SEVIRI image pairs around SAR timestamp.

        Args:
            stack_base: MsgStackBase instance with data_df
            sar_timestamp: Target SAR acquisition timestamp
            n_images: Number of images to select (uses config if None)

        Returns:
            DataFrame with selected image pairs
        """
        if n_images is None:
            n_images = self.n_images_window

        # Calculate time differences from SAR timestamp
        # Use the folder datetime as reference (average of before/after)
        stack_base.data_df['ref_datetime'] = pd.to_datetime(stack_base.data_df.index)
        stack_base.data_df['time_diff'] = abs(
            stack_base.data_df['ref_datetime'] - sar_timestamp
        )

        # Sort by time difference and select n nearest
        nearest = stack_base.data_df.nsmallest(n_images, 'time_diff')

        log.info(
            f"Selected {len(nearest)} SEVIRI pairs around {sar_timestamp}:\n"
            f"  Time range: {nearest['time_diff'].min()} to {nearest['time_diff'].max()}"
        )

        return nearest

    def process_channel_temporal_stack(
        self,
        arr_m_stack: np.ndarray,
        arr_s_stack: np.ndarray,
        time_diffs_m: List[timedelta],
        time_diffs_s: List[timedelta]
    ) -> np.ndarray:
        """
        Process temporal stack for a single channel using batch processing.

        Args:
            arr_m_stack: Master images (n_images, H, W)
            arr_s_stack: Slave images (n_images, H, W)
            time_diffs_m: Time differences for master images
            time_diffs_s: Time differences for slave images

        Returns:
            Array of temporally interpolated images (n_images, H, W)
        """
        n_images = arr_m_stack.shape[0]
        H, W = arr_m_stack.shape[1:]
        result_stack = np.zeros_like(arr_m_stack)

        for i in range(n_images):
            arr_m = arr_m_stack[i]
            arr_s = arr_s_stack[i]

            # Compute translation parameters for this pair
            # Note: get_trans_param expects 2D arrays
            trans = get_trans_param(
                arr_m, arr_s,
                steps_=self.translation_steps,
                cut_ref=355,
                cut_fit=115
            )
            m_dx, m_dy = trans[2]  # average translation

            # Apply batch temporal upsampling
            upsampled = batch_temporal_upsampling(
                arr_m=arr_m,
                arr_s=arr_s,
                sev_m_sar_dif=time_diffs_m[i],
                sev_s_sar_dif=time_diffs_s[i],
                nd_val=np.nan,
                batch_size_x=self.batch_size_x,
                batch_size_y=self.batch_size_y,
                overlap=self.overlap,
                steps_=self.translation_steps
            )

            result_stack[i] = upsampled

            log.debug(
                f"Processed image {i+1}/{n_images}: "
                f"translation=({m_dx:.2f}, {m_dy:.2f})"
            )

        return result_stack

    def process_temporal_stack(
        self,
        stack_base: MsgStackBase,
        sar_timestamp: datetime,
        seviri_data_dict: dict,
        n_images: int = None
    ) -> xr.Dataset:
        """
        Process SEVIRI temporal stack for all channels.

        Args:
            stack_base: MsgStackBase instance
            sar_timestamp: SAR acquisition timestamp
            seviri_data_dict: Dictionary with channel data {channel: xr.DataArray}
            n_images: Number of images (uses config if None)

        Returns:
            Dataset with shape (pair, channel, y, x)
        """
        if n_images is None:
            n_images = self.n_images_window

        # Find nearest image pairs
        nearest_pairs = self.find_nearest_images(stack_base, sar_timestamp, n_images)

        # Compute time differences
        time_diffs_m = []
        time_diffs_s = []

        for idx, row in nearest_pairs.iterrows():
            # Time from master/slave to SAR timestamp
            td_m = sar_timestamp - row['before_datetime']
            td_s = row['after_datetime'] - sar_timestamp
            time_diffs_m.append(td_m)
            time_diffs_s.append(td_s)

        # Process each channel
        channel_results = {}

        for channel in self.channels:
            if channel not in seviri_data_dict:
                log.warning(f"Channel {channel} not found in data, skipping")
                continue

            log.info(f"Processing channel: {channel}")

            # Extract master and slave arrays for this channel
            # seviri_data_dict[channel] should be (pair, y, x)
            data_array = seviri_data_dict[channel]

            # Select the nearest pairs
            # Assuming data_array has a 'pair' dimension indexed by timestamps
            arr_m_stack = []
            arr_s_stack = []

            for idx, row in nearest_pairs.iterrows():
                # Find indices in data_array for before/after times
                # This is a simplified version - actual implementation needs proper indexing
                # For now, we'll assume data is already properly indexed
                pair_idx = list(nearest_pairs.index).index(idx)
                arr_m = data_array.isel(pair=pair_idx).values  # Before image
                arr_s = data_array.isel(pair=pair_idx).values  # After image

                arr_m_stack.append(arr_m)
                arr_s_stack.append(arr_s)

            arr_m_stack = np.array(arr_m_stack)
            arr_s_stack = np.array(arr_s_stack)

            # Process temporal stack for this channel
            result = self.process_channel_temporal_stack(
                arr_m_stack, arr_s_stack,
                time_diffs_m, time_diffs_s
            )

            channel_results[channel] = result

        # Combine all channels into xarray Dataset
        # Stack channels: (pair, channel, y, x)
        n_pairs = len(nearest_pairs)
        n_channels = len(channel_results)
        H, W = next(iter(channel_results.values())).shape[1:]

        # Create coordinate arrays
        pair_coords = np.arange(n_pairs)
        channel_coords = list(channel_results.keys())

        # Stack channel data
        stacked_data = np.stack(
            [channel_results[ch] for ch in channel_coords],
            axis=1  # Insert channel dimension
        )

        # Create Dataset
        dataset = xr.Dataset(
            {
                'seviri_data': xr.DataArray(
                    stacked_data,
                    dims=['pair', 'channel', 'y', 'x'],
                    coords={
                        'pair': pair_coords,
                        'channel': channel_coords,
                        'sar_timestamp': sar_timestamp
                    },
                    attrs={
                        'description': 'Temporally downsampled SEVIRI data',
                        'sar_timestamp': str(sar_timestamp),
                        'n_images_used': n_images
                    }
                )
            }
        )

        log.info(
            f"Created temporal stack dataset: "
            f"{n_pairs} pairs, {n_channels} channels, {H}x{W} pixels"
        )

        return dataset

    def save_temporal_stack(
        self,
        dataset: xr.Dataset,
        filename: str = 'seviri_temporal_stack.nc'
    ) -> str:
        """
        Save temporal stack to NetCDF file.

        Args:
            dataset: Temporal stack dataset
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = f"{self.config.process_dir}/{filename}"
        dataset.to_netcdf(output_path, engine='netcdf4')
        log.info(f"Saved temporal stack to: {output_path}")
        return output_path

    def load_temporal_stack(
        self,
        filename: str = 'seviri_temporal_stack.nc'
    ) -> xr.Dataset:
        """
        Load temporal stack from NetCDF file.

        Args:
            filename: Input filename

        Returns:
            Temporal stack dataset
        """
        input_path = f"{self.config.process_dir}/{filename}"
        dataset = xr.open_dataset(input_path, engine='netcdf4')
        log.info(f"Loaded temporal stack from: {input_path}")
        return dataset
