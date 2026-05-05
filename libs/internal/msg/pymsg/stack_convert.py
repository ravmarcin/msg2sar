from .stack_base import MsgStackBase
from pathlib import Path
from tqdm import tqdm
import numpy as np
import traceback
import pyresample as pr
from pyresample.bilinear import NumpyBilinearResampler

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pyresample.kd_tree')


class MsgStackConvert(MsgStackBase):

    def __init__(
            self,
            data_dir: str,
            work_dir: str,
            geojson_path: str = None,
            projection: dict = None,
            resolution: float = 0.005,
        ) -> None:
        super().__init__(
            data_dir=data_dir,
            work_dir=work_dir,
            geojson_path=geojson_path,
            projection=projection,
            resolution=resolution
        )
        
    def _resample_values(self, values: np.ndarray, swath_def: pr.geometry.SwathDefinition,
                        area_def: pr.geometry.AreaDefinition, resample_method: str = "nearest",
                        **resample_kwargs):
        """
        Resample values using specified method.
        
        Parameters
        ----------
        values : np.ndarray
            Data values to resample
        swath_def : pyresample.geometry.SwathDefinition
            Source swath definition
        area_def : pyresample.geometry.AreaDefinition
            Target area definition
        resample_method : str
            Resampling method: 'nearest', 'gauss', or 'bilinear'
        **resample_kwargs
            Additional arguments for the resampling method
            
        Returns
        -------
        np.ndarray
            Resampled values
        """
        if resample_method == "nearest":
            radius = resample_kwargs.get('radius_of_influence', 1600)
            epsilon = resample_kwargs.get('epsilon', 0.5)
            fill_value = resample_kwargs.get('fill_value', False)
            return pr.kd_tree.resample_nearest(
                source_geo_def=swath_def, data=values, target_geo_def=area_def,
                radius_of_influence=radius, epsilon=epsilon, fill_value=fill_value
            )
        elif resample_method == "gauss":
            radius = resample_kwargs.get('radius_of_influence', 1600)
            sigmas = resample_kwargs.get('sigmas', 800)
            fill_value = resample_kwargs.get('fill_value', False)
            return pr.kd_tree.resample_gauss(
                source_geo_def=swath_def, data=values, target_geo_def=area_def,
                radius_of_influence=radius, sigmas=sigmas, fill_value=fill_value
            )
        elif resample_method == "bilinear":
            radius = resample_kwargs.get('radius_of_influence', 1600)
            epsilon = resample_kwargs.get('epsilon', 0.0)
            neighbours = resample_kwargs.get('neighbours', 21)
            reduce_data = resample_kwargs.get('reduce_data', True)
            segments = resample_kwargs.get('segments', None)
            
            resampler = NumpyBilinearResampler(
                source_geo_def=swath_def, target_geo_def=area_def,
                radius_of_influence=radius, neighbours=neighbours,
                reduce_data=reduce_data, epsilon=epsilon
            )
            return resampler.resample(data=values)
        else:
            raise ValueError(f"Unknown resample method: {resample_method}")

    def nat_to_tif_batch(self, dataset: str, area_def: pr.geometry.AreaDefinition, 
                         resample_method: str = "nearest",
                         epsg: int = 4326, no_data: float = -9999.0,
                         reader: str = "seviri_l1b_native", calibration: str = "radiance",
                         resample_kwargs: dict = None, overwrite: bool = False):
        """
        Convert NAT files in the DataFrame to GeoTIFF format.
        
        Iterates through all rows in self.data_df (which contains before_name and after_name files)
        and converts them from NAT format to GeoTIFF. Creates organized output structure in work_dir.
        
        Parameters
        ----------
        dataset : str
            Name of the dataset to extract from NAT files (e.g., 'IR_108', 'VIS006', etc.)
        area_def : pyresample.geometry.AreaDefinition, optional
            Area definition for resampling. If None and coordinates are provided,
            an AreaDefinition will be created.
        llx : float, optional
            Lower left x coordinate (required if area_def is None)
        lly : float, optional
            Lower left y coordinate (required if area_def is None)
        urx : float, optional
            Upper right x coordinate (required if area_def is None)
        ury : float, optional
            Upper right y coordinate (required if area_def is None)
        resample_method : str
            Resampling method: 'nearest' (default), 'gauss', or 'bilinear'
        epsg : int
            EPSG code for output projection. Default: 4326 (WGS84)
        no_data : float
            No data value for output. Default: -9999.0
        reader : str
            Satpy reader name. Default: "seviri_l1b_native"
        calibration : str
            Calibration type. Default: "radiance"
        resample_kwargs : dict, optional
            Additional parameters for resampling method
        overwrite : bool
            If True, overwrite existing output files. If False (default), skip existing files.
            
        Returns
        -------
        dict
            Dictionary with conversion statistics:
            - 'total': total number of files processed
            - 'converted': number of successfully converted files
            - 'skipped': number of files skipped (already exist and overwrite=False)
            - 'failed': list of files that failed conversion
            - 'output_dir': path to the output directory
            
        Raises
        ------
        ValueError
            If area_def is None and coordinates are not provided
        """
        
        # Default resample kwargs
        if resample_kwargs is None:
            resample_kwargs = {}
        
        # Create output directory structure
        output_base = Path(self.work_dir) / "nat_to_tiff" / dataset
        output_base.mkdir(parents=True, exist_ok=True)
        
        # Track conversion results
        results = {
            'total': 0,
            'converted': 0,
            'skipped': 0,
            'failed': [],
            'output_dir': str(output_base)
        }
        
        # Find all subdirectories in data_dir (these contain the unpacked NAT files)
        data_path = Path(self.data_dir)
        subdirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
        
        # Create a mapping of filenames to their full paths
        # Only look for .nat files and skip zip files
        filename_to_path = {}
        for subdir in subdirs:
            for file in subdir.rglob('*.nat'):
                if file.is_file() and file.suffix == '.nat':
                    filename_to_path[str(file.name)[:-4]] = str(file)
        
        # Process each row in the DataFrame
        for idx, row in tqdm(self.data_df.iterrows(), total=len(self.data_df), desc=f"Converting NAT to TIFF ({dataset})"):
            try:
                # Create subdirectory based on row index (folder_datetime)
                row_index_str = idx.strftime('%Y%m%d_%H%M%S') if hasattr(idx, 'strftime') else str(idx)
                output_row_dir = output_base / row_index_str
                output_row_dir.mkdir(parents=True, exist_ok=True)
                
                # Process before_name file
                before_name = row['before_name']
                if before_name in filename_to_path:
                    before_path = filename_to_path[before_name]
                    
                    try:
                        results['total'] += 1
                        output_file = output_row_dir / f"{Path(before_path).stem}_{dataset}.tif"
                        
                        # Check if output already exists
                        if output_file.exists() and not overwrite:
                            results['skipped'] += 1
                            continue
                        
                        # Load scene and dataset
                        scn = self._get_scene_dataset(before_path, dataset, reader, calibration)
                        # Get resampling components
                        values = scn[dataset].values.astype(np.float32)
                        swath_def = self._get_swath_definition(scn, dataset)
                        
                        # Resample values
                        resampled = self._resample_values(values, swath_def, area_def, 
                                                         resample_method, **resample_kwargs)
                        
                        # Write to GeoTIFF
                        super()._write_geotiff(str(output_file), resampled, area_def, epsg, no_data)
                        results['converted'] += 1
                    except Exception as e:
                        error_msg = f"{str(e)}\n{traceback.format_exc()}"
                        results['failed'].append({
                            'file': before_name,
                            'error': error_msg
                        })
                else:
                    results['failed'].append({
                        'file': before_name,
                        'error': 'File not found in data_dir'
                    })
                
                # Process after_name file
                after_name = row['after_name']
                if after_name in filename_to_path:
                    after_path = filename_to_path[after_name]
                    try:
                        results['total'] += 1
                        output_file = output_row_dir / f"{Path(after_path).stem}_{dataset}.tif"
                        
                        # Check if output already exists
                        if output_file.exists() and not overwrite:
                            results['skipped'] += 1
                            continue
                        
                        # Load scene and dataset
                        scn = self._get_scene_dataset(after_path, dataset, reader, calibration)
                        
                        # Get resampling components
                        values = scn[dataset].values.astype(np.float32)
                        swath_def = self._get_swath_definition(scn, dataset)
                        
                        # Resample values
                        resampled = self._resample_values(values, swath_def, area_def,
                                                         resample_method, **resample_kwargs)
                        
                        # Write to GeoTIFF
                        super()._write_geotiff(str(output_file), resampled, area_def, epsg, no_data)
                        results['converted'] += 1
                    except Exception as e:
                        error_msg = f"{str(e)}\n{traceback.format_exc()}"
                        results['failed'].append({
                            'file': after_name,
                            'error': error_msg
                        })
                else:
                    results['failed'].append({
                        'file': after_name,
                        'error': 'File not found in data_dir'
                    })
                    
            except Exception as e:
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                results['failed'].append({
                    'row_index': str(idx),
                    'error': error_msg
                })
        
        return results