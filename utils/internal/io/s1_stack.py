from pygmtsar import S1, Stack
import geopandas as gpd



def init_stack(
        aoi: gpd.GeoDataFrame,
        ref: str,
        data_dir: str,
        work_dir: str,
        verbose: bool = True,
        drop_if_exists: bool = False,
        dem: str = None,
    ) -> Stack:
    
    if verbose:
        print(f"Scan data directory for S1 scenes: {data_dir}")
    scenes = S1.scan_slc(data_dir)
    
    if verbose:
        print(f"Initialize S1 Stack with working directory: {work_dir}")
    sbas = Stack(work_dir, drop_if_exists=drop_if_exists)
    sbas.reference = ref
    sbas = sbas.set_scenes(scenes)
    sbas = sbas.set_reference(ref)
    

    if dem is not None:
        if verbose:
            print(f"Load DEM from {dem}")
        sbas.load_dem(dem, aoi)
    
    return sbas