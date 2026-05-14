# GNSS CDS Module

Downloads and processes GNSS Zenith Tropospheric Delay (ZTD) and Total Column Water Vapour (TCWV) data from the Copernicus Climate Data Store (CDS) for InSAR atmospheric correction validation.

## Data Source

**Dataset:** `insitu-observations-gnss` on Copernicus CDS
**URL:** https://cds.climate.copernicus.eu/datasets/insitu-observations-gnss
**Coverage:** IGS and EPN networks, 1996–present
**Variables:** ZTD, TCWV, TCWV (ERA5 comparison)

---

## Modules

### `gnss_config.py`

Configuration loader following the project's config pattern.

**Class:** `GnssConfig`

Loads from JSON config and provides:
| Attribute | Description |
|-----------|-------------|
| `gnss_stations` | Station selection (`"auto"` or list) |
| `temporal_buffer_hours` | Time window for temporal filtering (default: 6h) |
| `spatial_buffer_km` | Spatial buffer around AOI (default: 100km) |
| `cds_network_type` | Network type (`epn_repro2`, `igs_repro3`, or auto-select) |
| `cds_variables` | Variables to download (default: `["zenith_total_delay"]`) |
| `year` | Optional year override for full-year downloads |
| `dates` | Optional list of specific dates (`YYYYMMDD` format) |
| `download_dir` | Resolved output directory |
| `aoi` | GeoDataFrame of the buffered area of interest |

---

### `gnss_downloader.py`

Downloads GNSS ZTD/TCWV data from the CDS API.

**Class:** `GnssDownloader`

**Key methods:**
| Method | Description |
|--------|-------------|
| `download_gnss_stations(aoi, date_range)` | Download from CDS API for an AOI and date range |
| `load_gnss_from_file(filepath, format)` | Load from local file (CSV, Nevada, IGS formats) |
| `compute_reference_ztd(sar_timestamp, ref_point)` | Interpolated ZTD at reference point for InSAR correction |

**Download features:**
- Monthly chunking to avoid CDS API size limits
- Supports specific dates, full year, or date range downloads
- Bounding box spatial filtering (CDS format: N, W, S, E)
- Auto-selects network: EPN for Europe, IGS for global
- Multi-variable support with pivot to wide format
- Credentials from `.secrets/keys.json` or `~/.cdsapirc`

**Supported file formats for `load_gnss_from_file`:**
- `csv` — Standard CSV with columns: `station_id, lat, lon, ztd, datetime`
- `nevada` — Nevada Geodetic Laboratory whitespace-delimited format
- `igs` — IGS ZTD format (space-separated: datetime, ztd, lat, lon)

---

### `gnss_processor.py`

Spatial and temporal interpolation of GNSS ZTD data to match InSAR geometry.

**Class:** `GnssProcessor`

**Key methods:**
| Method | Description |
|--------|-------------|
| `load_gnss_data()` | Load GNSS data from CSV |
| `interpolate_ztd_spatial(lat, lon, timestamp)` | Spatial interpolation to target location |
| `interpolate_ztd_temporal(station_id, timestamps)` | Temporal interpolation for a station |
| `compute_ztd_grid(sar_grid, timestamp)` | Compute ZTD on a full SAR grid |
| `save_ztd_grid(ztd_grid, filename)` | Save to NetCDF |

**Interpolation details:**
- Spatial: `scipy.interpolate.griddata` (linear, cubic, or nearest)
- Temporal: `scipy.interpolate.interp1d` (linear with extrapolation)
- Falls back to nearest-station value when fewer than 3 stations available
- Averaging over configurable time window per station before spatial interpolation

---

## Processing Script

`scripts/download_cds_data.py` — CLI interface for downloading GNSS ZTD/TCWV data.

```bash
# Download from Copernicus CDS (requires credentials)
python scripts/download_cds_data.py --config data/configs/gnss/2022/bogo_pl.json --method api

# Load from pre-downloaded CSV
python scripts/download_cds_data.py --config data/configs/gnss/2022/bogo_pl.json --method file --file gnss_data.csv

# Show manual download instructions
python scripts/download_cds_data.py --help-manual
```

---

## Dependencies

- `cdsapi` — Copernicus Climate Data Store API client
- `geopandas` — AOI geometry handling
- `scipy` — Spatial/temporal interpolation (`griddata`, `interp1d`)
- `xarray` — Grid output (NetCDF)
- `pandas`, `numpy` — Data handling

---

## Credentials Setup

CDS API credentials are required for downloading. Two options:

**Option 1:** Add to `.secrets/keys.json`:
```json
{
  "cdsapi": {
    "url": "https://cds.climate.copernicus.eu/api",
    "token": "YOUR_UID:YOUR_API_KEY"
  }
}
```

**Option 2:** Create `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api
key: YOUR_UID:YOUR_API_KEY
```

Get credentials from: https://cds.climate.copernicus.eu/user

---

## Config Example

```json
{
  "job_name": "gnss_ztd_poland",
  "data": {
    "aoi": {
      "polygon_child_dir": "aoi/poland.geojson",
      "name": "poland"
    },
    "gnss": {
      "data_child_dir": "gnss/cds/2021",
      "download_folder": "downloads",
      "output_name": "gnss_ztd.csv",
      "stations": "auto",
      "temporal_buffer_hours": 6,
      "spatial_buffer_km": 100,
      "cds_network_type": "epn_repro2",
      "cds_variables": ["zenith_total_delay", "total_column_water_vapour"],
      "dates": ["20210410", "20210411"]
    }
  }
}
```

---

## Usage

```python
from libs.internal.gnss.cds import GnssDownloader, GnssProcessor, GnssConfig

# Download ZTD data
downloader = GnssDownloader("path/to/config.json")
data = downloader.download_gnss_stations(aoi_gdf, (start_date, end_date))

# Or load from existing file
data = downloader.load_gnss_from_file("gnss_ztd.csv")

# Process: interpolate ZTD to InSAR grid
processor = GnssProcessor("path/to/config.json")
processor.load_gnss_data()
ztd_value = processor.interpolate_ztd_spatial(52.47, 21.03, sar_timestamp)
ztd_grid = processor.compute_ztd_grid(sar_grid, sar_timestamp)
processor.save_ztd_grid(ztd_grid, "ztd_correction.nc")
```
