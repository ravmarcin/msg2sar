# GNSS RINEX Processing Module

Downloads and processes RINEX 3 observation and navigation files for GNSS-based positioning and tropospheric delay estimation.

## Modules

### `rinex_downloader.py`

Downloads RINEX 3 files from the BKG GNSS Data Center (`https://igs.bkg.bund.de`).

**Classes:**
- `RinexDownloader` — Main downloader class
- `RinexFile` — Parsed RINEX 3 long filename metadata (dataclass)

**Key methods:**
| Method | Description |
|--------|-------------|
| `list_files(date, station, data_type)` | List available RINEX files for a given date |
| `download_file(date, filename)` | Download a single RINEX file |
| `download_station(station, date, obs, nav)` | Download all files for a station on a date |
| `download_date_range(station, start, end)` | Download over a date range |
| `download_stations(stations, date)` | Download for multiple stations on a single date |

**Supported networks:** IGS, EUREF, MGEX, GREF, IGLOS, EPNrepro1, EPNrepro2, BfGNet, IGSac, MISC, NTRIP

**Helper functions:**
- `parse_rinex_filename(filename)` — Parse RINEX 3 long filename into components
- `date_to_doy(date)` — Convert datetime to (year, day-of-year)

---

### `orbit_downloader.py`

Downloads precise orbit (SP3) and clock files from the GFZ data center for orbit interpolation.

**Source:** `ftp://ftp.gfz-potsdam.de/pub/GNSS/products/final/wWWWW/`

**Key functions:**
| Function | Description |
|----------|-------------|
| `download_orbits(epoch, output_dir, product)` | Download 3 SP3 + 1 CLK file for a date |
| `download_orbits_for_range(start, end, output_dir)` | Download orbit files for a date range |

**File naming:** `{product}{gps_week}{weekday}.{sp3|clk}.Z` (e.g., `gfz21526.sp3.Z`)

Files needed for `sp3_interp` (Lagrange orbit interpolation):
- SP3: yesterday, today, tomorrow (3 files)
- CLK: today (1 file)

---

### `rinex_processor.py`

Processes downloaded RINEX files using the `gnsspy` library.

**Class:** `GnssRinexProcessor`

**Key methods:**
| Method | Description |
|--------|-------------|
| `read_observation(filename)` | Read RINEX observation file (handles .gz + .crx decompression) |
| `read_navigation(filename)` | Read RINEX navigation file |
| `compute_spp_with_orbits(obs, orbit_dir, ...)` | Per-epoch SPP with precise orbits (main method) |
| `compute_spp(obs, orbit, system, cut_off)` | Basic SPP via gnsspy (averaged solution) |
| `compute_multipath(obs, system)` | Multipath indicators (MP1, MP2) |
| `compute_tropospheric_delay(obs, elevation)` | Collins (1999) zenith tropospheric delay |
| `interpolate_orbits(epoch, interval, product)` | Lagrange interpolation of precise orbits |
| `get_station_position(obs)` | ECEF to geodetic conversion from RINEX header |
| `get_observation_summary(obs)` | File metadata and observation statistics |
| `list_rinex_files()` | List files in data directory grouped by type |

#### `compute_spp_with_orbits` — SPP Pipeline

The primary processing method. Produces per-epoch (30s) positions with observation-derived tropospheric correction:

1. Reads RINEX observation file
2. Interpolates precise orbits from SP3/CLK files (Lagrange)
3. Builds ionosphere-free pseudorange combination (dual-frequency)
4. Computes per-epoch least-squares positions with:
   - Satellite positions corrected for signal travel time and Sagnac effect
   - Tropospheric correction via Collins model using actual satellite elevation angles
   - Elevation cutoff filtering (default 7 deg)

**Output columns:** `Epoch, X, Y, Z, receiver_clock, n_sats, tropo_mean_m`

**Notes:**
- Uses `station.approx_position` (static a priori from RINEX header) for geometry (elevation, azimuth, tropo mapping) — standard practice
- Uses iteratively-updated position for the linearized design matrix
- `tropo_mean_m` is the mean **slant** tropospheric delay across visible satellites (not ZTD)
- Requires `hatanaka` package for CRX to RNX decompression on macOS

---

## Processing Script

`scripts/process_rinex_data.py` — CLI interface for the full pipeline.

```bash
# List available RINEX files
python scripts/process_rinex_data.py --config <config.json> --list

# Export 30s observations + daily station summary
python scripts/process_rinex_data.py --config <config.json> --summary

# SPP with precise orbits (downloads SP3/CLK from GFZ)
python scripts/process_rinex_data.py --config <config.json> --spp

# Multipath analysis
python scripts/process_rinex_data.py --config <config.json> --multipath

# All processing steps
python scripts/process_rinex_data.py --config <config.json> --all
```

**Output files** (in `data/.../processed/`):
| File | Content |
|------|---------|
| `{STATION}_{DATE}_obs.csv` | 30s observations (all satellites, all signals) |
| `{STATION}_{DATE}_spp.csv` | Per-epoch SPP positions + tropospheric delay |
| `{STATION}_{DATE}_multipath.csv` | Multipath indicators per satellite per epoch |
| `station_daily_summary.csv` | Daily metadata with SPP-derived position and tropo stats |

---

## Dependencies

- `gnsspy` — RINEX reading, orbit interpolation, SPP internals
- `hatanaka` — Cross-platform CRX to RNX decompression (replaces gnsspy's Linux-only binary)
- `pandas`, `numpy` — Data handling
- `requests` — RINEX file downloads from BKG

**Known gnsspy compatibility patches** (pandas 2.x):
- `interpolation.py`: `freq = str(interval) + 'S'` must be lowercase `'s'`
- `station.receiver_clock`: parsed as string from RINEX header, must be cast to float

---

## Config Example

```json
{
  "data": {
    "rinex": {
      "data_child_dir": "gnss/rinex/2021/bogo_pl",
      "download_folder": "rinex_data"
    }
  }
}
```
