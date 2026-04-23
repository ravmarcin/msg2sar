# ----------------------------------------------------------------------------
# PyGMTSAR
# 
# This file is part of the PyGMTSAR project: https://github.com/mobigroup/gmtsar
# 
# Copyright (c) 2021, Alexey Pechnikov
# 
# Licensed under the BSD 3-Clause License (see LICENSE for details)
# ----------------------------------------------------------------------------
from pathlib import Path
from datetime import datetime
import re
import json
import pandas as pd
import zipfile
from tqdm import tqdm
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import pyresample as pr
from pyresample.bilinear import NumpyBilinearResampler
from satpy import Scene

from utils.external.pygmtsar.IO import IO
from utils.external.pygmtsar.tqdm_joblib import tqdm_joblib


class MsgStackBase(tqdm_joblib, IO):

    def __init__(
            self,
            data_dir: str,
            work_dir: str,
            geojson_path: str,
            projection: dict, 
            resolution: float = 0.005
        ) -> None:
        super().__init__()
        self.data_df = pd.DataFrame()
        self.data_dir = data_dir
        self.work_dir = work_dir
        self.geojson_path = geojson_path
        self.resolution = resolution
        self.projection = projection
        self.read_data_dir(self.data_dir)
        self.unpack_data_dir(self.data_dir, skip_if_exists=True)
        self.area_def = self.get_area_def_from_geojson(
            self.geojson_path,
            spatial_buffer_meters=0,
            resolution=self.resolution,
            projection=projection
        )

    def __repr__(self):
        return 'Object %s %d items\n%r' % (self.__class__.__name__, len(self.data_df), self.data_df)
    
    def read_data_dir(self, archive_path):
        """
        Read archive directory structure and create a DataFrame with file pairs.
        
        The function looks for subdirectories with datetime-based names (YYYYMMDD_HHMMSS format)
        and files with embedded datetimes. It pairs consecutive files by their datetime stamps.
        
        Parameters
        ----------
        archive_path : str or Path
            Path to the archive directory containing subdirectories with timestamp names.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with index as folder datetimes and columns:
            - before_name : str
                Name of the earlier file
            - before_datetime : datetime
                Datetime extracted from the earlier file
            - after_name : str
                Name of the later file
            - after_datetime : datetime
                Datetime extracted from the later file
                
        Examples
        --------
        df = stack.read_archive(
            "/Users/raf/Dev/code/msg2sar/data/msg/seviri/2023/bogo_pl/download"
        )
        """
        archive_path = Path(archive_path)
        
        # Pattern to extract datetime from filenames (YYYYMMDDHHMMSS format)
        # Looks for 14 consecutive digits in the filename
        datetime_pattern = re.compile(r'(\d{14})')
        
        rows = []
        
        # Get all subdirectories sorted
        subdirs = sorted([d for d in archive_path.iterdir() if d.is_dir()])
        
        for subdir in subdirs:
            # Parse folder name as index datetime (YYYYMMDD_HHMMSS or YYYYMMDDHHMMSS)
            folder_name = subdir.name
            try:
                # Try parsing with underscore separator
                if '_' in folder_name:
                    idx_datetime = datetime.strptime(folder_name, '%Y%m%d_%H%M%S')
                else:
                    idx_datetime = datetime.strptime(folder_name, '%Y%m%d%H%M%S')
            except ValueError:
                # Skip if folder name doesn't match expected format
                continue
            
            # Get all files in subdirectory
            files = sorted([f for f in subdir.iterdir() if f.is_file()])
            
            # Extract datetime from each file
            file_datetimes = []
            for file in files:
                match = datetime_pattern.search(file.name)
                if match:
                    dt_str = match.group(1)
                    try:
                        dt = datetime.strptime(dt_str, '%Y%m%d%H%M%S')
                        # Remove .zip extension if present
                        file_name = file.name
                        if file_name.endswith('.zip'):
                            file_name = file_name[:-4]
                        file_datetimes.append({
                            'name': file_name,
                            'datetime': dt,
                            'path': str(file)
                        })
                    except ValueError:
                        continue
            
            # Sort by datetime
            file_datetimes.sort(key=lambda x: x['datetime'])
            
            # Pair consecutive files (before and after)
            for i in range(len(file_datetimes) - 1):
                before = file_datetimes[i]
                after = file_datetimes[i + 1]
                
                rows.append({
                    'before_name': before['name'],
                    'before_datetime': before['datetime'],
                    'after_name': after['name'],
                    'after_datetime': after['datetime']
                })
        
        # Create DataFrame with folder datetime as index
        if rows:
            self.data_df = pd.DataFrame(rows)
            # Add index based on folder names
            self.data_df.index = [
                datetime.strptime(
                    subdir.name if '_' not in subdir.name else subdir.name.replace('_', ''),
                    '%Y%m%d%H%M%S'
                )
                for subdir in sorted([d for d in archive_path.iterdir() if d.is_dir()])
            ][:len(self.data_df)]
            self.data_df.index.name = 'folder_datetime'
        else:
            self.data_df = pd.DataFrame(columns=[
                'before_name', 'before_datetime', 'after_name', 'after_datetime'
            ])
            self.data_df.index.name = 'folder_datetime'



    def unpack_file(self, zip_path, output_dir=None, skip_if_exists=True):
        """
        Unpack a single zip file into a folder with the same name (without .zip extension).
        
        Parameters
        ----------
        zip_path : str or Path
            Path to the zip file to unpack.
        output_dir : str or Path, optional
            Directory to extract the zip file to. If None, extracts to a folder
            with the same name as the zip file (without .zip extension) in the same directory.
        skip_if_exists : bool, optional
            If True (default), skip unpacking if the output directory already exists and is not empty.
            
        Returns
        -------
        Path
            Path to the extracted directory.
            
        Examples
        --------
        output_path = stack.unpack_file(
            "/path/to/archive/file.zip"
        )
        """
        zip_path = Path(zip_path)
        
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
        
        if not zip_path.suffix.lower() == '.zip':
            raise ValueError(f"File is not a zip archive: {zip_path}")
        
        # If output_dir is not specified, create folder with same name as zip (without .zip)
        if output_dir is None:
            output_dir = zip_path.parent / zip_path.stem
        else:
            output_dir = Path(output_dir)
        
        # Check if output directory already exists and has content
        if skip_if_exists and output_dir.exists() and any(output_dir.iterdir()):
            return output_dir
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        return output_dir

    def unpack_data_dir(self, data_dir=None, skip_if_exists=True):
        """
        Unpack all zip files in the data directory structure.
        
        Each zip file is extracted into a folder with the same name as the zip file
        (without the .zip extension) in the same directory.
        
        Parameters
        ----------
        data_dir : str or Path, optional
            Path to the data directory. If None, uses self.data_dir.
        skip_if_exists : bool, optional
            If True (default), skip unpacking if the output directory already exists and is not empty.
            
        Returns
        -------
        dict
            Dictionary with three keys:
            - 'extracted': list of paths that were extracted
            - 'skipped': list of paths that were skipped (already exist)
            - 'failed': dictionary of paths and their error messages
            
        Examples
        --------
        result = stack.unpack_archive(
            "/Users/raf/Dev/code/msg2sar/data/msg/seviri/2023/bogo_pl/download"
        )
        print(f"Extracted: {len(result['extracted'])} files")
        print(f"Skipped: {len(result['skipped'])} files")
        """
        if data_dir is None:
            data_dir = self.data_dir
        
        data_dir = Path(data_dir)
        result = {
            'extracted': [],
            'skipped': [],
            'failed': {}
        }
        
        # Find all zip files in the data directory
        zip_files = sorted(data_dir.rglob('*.zip'))
        
        # Iterate through zip files with tqdm progress bar
        for zip_file in tqdm(zip_files, desc="Unpacking archives", unit="file"):
            output_dir = zip_file.parent / zip_file.stem
            
            # Check if already extracted
            if skip_if_exists and output_dir.exists() and any(output_dir.iterdir()):
                result['skipped'].append(str(zip_file))
                continue
            
            try:
                self.unpack_file(zip_file, skip_if_exists=skip_if_exists)
                result['extracted'].append(str(zip_file))
            except Exception as e:
                result['failed'][str(zip_file)] = str(e)
        
        return result

    def read_work_dir(self, work_path=None):
        """
        Read work directory structure and create a DataFrame with processed files.
        
        Similar to read_data_dir, this function reads subdirectories with datetime-based names
        and creates pairs of files for differential processing.
        
        Parameters
        ----------
        work_path : str or Path, optional
            Path to the work directory. If None, uses self.work_dir.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with index as folder datetimes and columns for file pairs
        """
        if work_path is None:
            work_path = self.work_dir
        
        work_path = Path(work_path)
        
        if not work_path.exists():
            self.df = pd.DataFrame(columns=['before_name', 'before_datetime', 'after_name', 'after_datetime'])
            self.df.index.name = 'folder_datetime'
            return self.df
        
        # Pattern to extract datetime from filenames (YYYYMMDDHHMMSS format)
        datetime_pattern = re.compile(r'(\d{14})')
        
        rows = []
        
        # Get all subdirectories sorted
        subdirs = sorted([d for d in work_path.iterdir() if d.is_dir()])
        
        for subdir in subdirs:
            # Parse folder name as index datetime (YYYYMMDD_HHMMSS or YYYYMMDDHHMMSS)
            folder_name = subdir.name
            try:
                # Try parsing with underscore separator
                if '_' in folder_name:
                    idx_datetime = datetime.strptime(folder_name, '%Y%m%d_%H%M%S')
                else:
                    idx_datetime = datetime.strptime(folder_name, '%Y%m%d%H%M%S')
            except ValueError:
                # Skip if folder name doesn't match expected format
                continue
            
            # Get all files in subdirectory
            files = sorted([f for f in subdir.iterdir() if f.is_file()])
            
            # Extract datetime from each file
            file_datetimes = []
            for file in files:
                match = datetime_pattern.search(file.name)
                if match:
                    dt_str = match.group(1)
                    try:
                        dt = datetime.strptime(dt_str, '%Y%m%d%H%M%S')
                        # Remove .zip extension if present
                        file_name = file.name
                        if file_name.endswith('.zip'):
                            file_name = file_name[:-4]
                        file_datetimes.append({
                            'name': file_name,
                            'datetime': dt,
                            'path': str(file)
                        })
                    except ValueError:
                        continue
            
            # Sort by datetime
            file_datetimes.sort(key=lambda x: x['datetime'])
            
            # Pair consecutive files (before and after)
            for i in range(len(file_datetimes) - 1):
                before = file_datetimes[i]
                after = file_datetimes[i + 1]
                
                rows.append({
                    'before_name': before['name'],
                    'before_datetime': before['datetime'],
                    'after_name': after['name'],
                    'after_datetime': after['datetime']
                })
        
        # Create DataFrame with folder datetime as index
        if rows:
            self.df = pd.DataFrame(rows)
            # Add index based on folder names
            self.df.index = [
                datetime.strptime(
                    subdir.name if '_' not in subdir.name else subdir.name.replace('_', ''),
                    '%Y%m%d%H%M%S'
                )
                for subdir in sorted([d for d in work_path.iterdir() if d.is_dir()])
            ][:len(self.df)]
            self.df.index.name = 'folder_datetime'
        else:
            self.df = pd.DataFrame(columns=['before_name', 'before_datetime', 'after_name', 'after_datetime'])
            self.df.index.name = 'folder_datetime'
        
        return self.df

    def _get_scene_dataset(self, file: str, dataset: str, reader: str = "seviri_l1b_native", 
                          calibration: str = "radiance"):
        """
        Load a dataset from a NAT file using satpy.
        
        Parameters
        ----------
        file : str
            Path to NAT file
        dataset : str
            Dataset name to load
        reader : str
            Satpy reader name
        calibration : str
            Calibration type
            
        Returns
        -------
        satpy.Scene
            Scene object with loaded dataset
        """
        scn = Scene(filenames={reader: [file]})
        scn_names = scn.all_dataset_names()
        if dataset in scn_names:
            scn.load([dataset], calibration=calibration)
            return scn
        else:
            raise ValueError(f"Dataset '{dataset}' not available in file. Available: {scn_names}")

    def _get_swath_definition(self, scn: Scene, dataset: str):
        """
        Get swath definition from scene for resampling.
        
        Parameters
        ----------
        scn : satpy.Scene
            Scene object with loaded dataset
        dataset : str
            Dataset name
            
        Returns
        -------
        pyresample.geometry.SwathDefinition
            Swath definition for the dataset
        """
        lons, lats = scn[dataset].area.get_lonlats()
        swath_def = pr.geometry.SwathDefinition(lons=lons, lats=lats)
        return swath_def

    def _write_geotiff(self, output_path: str, values: np.ndarray, area_def: pr.geometry.AreaDefinition,
                      epsg: int = 4326, no_data: float = -9999.0):
        """
        Write resampled values to GeoTIFF file using rasterio.
        
        Parameters
        ----------
        output_path : str
            Output file path
        values : np.ndarray
            Data values to write
        area_def : pyresample.geometry.AreaDefinition
            Area definition with georeferencing info
        epsg : int
            EPSG code for projection
        no_data : float
            No data value
        """
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Get extent from area_def
        minx, miny, maxx, maxy = area_def.area_extent
        
        # Create transform from bounds
        transform = from_bounds(minx, miny, maxx, maxy, values.shape[1], values.shape[0])
        
        # Write GeoTIFF using rasterio
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=values.shape[0],
            width=values.shape[1],
            count=1,
            dtype=values.dtype,
            crs=f'EPSG:{epsg}',
            transform=transform,
            nodata=no_data
        ) as dst:
            dst.write(values.astype(np.float32), 1)

    def get_area_def_from_geojson(
            self,
            geojson_path: str,
            projection: dict,
            spatial_buffer_meters: float = 0, 
            resolution: float = 0.005, 
            epsg: int = 4326
        ):
        """
        Create a pyresample AreaDefinition directly from a GeoJSON polygon file.
        
        Extracts the bounding box from the polygon geometry, applies spatial buffering,
        and creates an area definition with the specified projection and resolution.
        
        Parameters
        ----------
        geojson_path : str or Path
            Path to the GeoJSON file (e.g., bogo_pl.geojson)
        spatial_buffer_meters : float, optional
            Spatial buffer to apply in meters (default: 0)
        resolution : float, optional
            Resolution in degrees (default: 0.005)
        epsg : int, optional
            EPSG code for projection (default: 4326)
            
        Returns
        -------
        pyresample.geometry.AreaDefinition
            Area definition with spatial buffer applied
            
        Raises
        ------
        FileNotFoundError
            If GeoJSON file not found
        ValueError
            If GeoJSON structure is invalid
            
        Examples
        --------
        area_def = stack.get_area_def_from_geojson(
            "/Users/raf/Dev/code/msg2sar/data/polygons/bogo_pl.geojson",
            spatial_buffer_meters=4000,
            resolution=0.005
        )
        """
        geojson_path = Path(geojson_path)
        
        if not geojson_path.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
        
        # Load GeoJSON
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        # Extract coordinates from GeoJSON
        bbox_coords = {}
        if 'features' in geojson_data and len(geojson_data['features']) > 0:
            geometry = geojson_data['features'][0]['geometry']
            
            if geometry['type'] == 'MultiPolygon':
                coords = geometry['coordinates'][0][0]  # First polygon, first ring
            elif geometry['type'] == 'Polygon':
                coords = geometry['coordinates'][0]  # First ring
            else:
                raise ValueError(f"Unsupported geometry type: {geometry['type']}")
            
            # Extract all lon/lat pairs
            lons = [coord[0] for coord in coords]
            lats = [coord[1] for coord in coords]
            
            bbox_coords = {
                'minx': min(lons),
                'maxx': max(lons),
                'miny': min(lats),
                'maxy': max(lats)
            }
        else:
            raise ValueError("No features found in GeoJSON file")
        
        # Apply spatial buffer (convert meters to degrees, 1 degree ~= 111 km)
        buffer_degrees = spatial_buffer_meters / 111000.0
        
        minx = bbox_coords['minx'] - buffer_degrees
        maxx = bbox_coords['maxx'] + buffer_degrees
        miny = bbox_coords['miny'] - buffer_degrees
        maxy = bbox_coords['maxy'] + buffer_degrees
        
        # Get AOI name from GeoJSON if available
        aoi_name = 'aoi'
        if 'features' in geojson_data and len(geojson_data['features']) > 0:
            feature = geojson_data['features'][0]
            if 'properties' in feature and 'aoiName' in feature['properties']:
                aoi_name = feature['properties']['aoiName']
        
        # Create area definition using EPSG code
        area_def = pr.geometry.AreaDefinition(
            area_id=aoi_name,
            description=f"Area definition for {aoi_name} with buffer {spatial_buffer_meters}m",
            proj_id=f'EPSG:{epsg}',
            projection=projection,
            width=int((maxx - minx) / resolution),
            height=int((maxy - miny) / resolution),
            area_extent=(minx, miny, maxx, maxy)
        )
        
        return area_def

    def to_dataframe(self):
        """
        Return a Pandas DataFrame for all Stack scenes.

        Returns
        -------
        pandas.DataFrame
            The DataFrame containing Stack scenes.

        Examples
        --------
        df = stack.to_dataframe()
        """
        return self.data_df