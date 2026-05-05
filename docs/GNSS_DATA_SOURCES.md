# GNSS Data Sources Guide

Guide for obtaining GNSS tropospheric delay data for atmospheric correction validation.

---

## Quick Start

### Option 1: Automated Download (Recommended) ⭐

Use the Copernicus Climate Data Store API for automated downloads:

```bash
# First-time setup: Configure CDS API credentials (see below)

# Download GNSS data automatically
python scripts/download_gnss_data.py \
    --config data/configs/gnss/2023/bogo_pl.json \
    --method api
```

**Requirements:**
- Free CDS account (register at https://cds.climate.copernicus.eu/user/register)
- CDS API credentials configured in `~/.cdsapirc`
- `cdsapi` package installed (`pip install cdsapi`)

### Option 2: Generate Test Data

For initial testing without API setup:

```bash
python scripts/download_gnss_data.py \
    --config data/configs/gnss/2023/bogo_pl.json \
    --method synthetic
```

### Option 3: Manual Download

For specific requirements, manually download from sources below.

---

## GNSS Data Sources

### 1. Copernicus Climate Data Store (Automated) ⭐

**Website:** https://cds.climate.copernicus.eu/datasets/insitu-observations-gnss

**Coverage:** Global (IGS) and European (EPN) networks, 1996-present

**Pros:**
- **Automated API access** - No manual download needed
- High-quality reprocessed data (IGS-repro3, EPN-repro2)
- Consistent data format (CSV or NetCDF)
- Hourly ZTD measurements
- Free access with registration

**Setup:**

1. **Register for free account:**
   - Go to https://cds.climate.copernicus.eu/user/register
   - Verify your email

2. **Get API credentials:**
   - Log in and go to your user profile
   - Copy your UID and API key

3. **Configure credentials:**
   Create `~/.cdsapirc` file:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: YOUR_UID:YOUR_API_KEY
   ```

4. **Install Python package:**
   ```bash
   pip install cdsapi
   ```

**Usage:**

```bash
# Automatic download via API
python scripts/download_gnss_data.py \
    --config data/configs/gnss/2023/bogo_pl.json \
    --method api
```

**Networks Available:**
- `igs_repro3`: Global IGS network reprocessing (500+ stations)
- `epn_repro2`: European EUREF network reprocessing (300+ stations)
- `igs_daily`: Near real-time IGS daily products

**Data Format:**
- ZTD (Zenith Total Delay) in meters
- TCWV (Total Column Water Vapour) in kg/m²
- Hourly temporal resolution
- CSV format

**Features:**
- Automatically selects appropriate network based on region (EPN for Europe, IGS for global)
- Splits large requests into monthly chunks to avoid API limits
- Handles errors gracefully - continues if some months fail
- Combines all downloaded data automatically

---

### 2. Nevada Geodetic Laboratory (Manual)

**Website:** http://geodesy.unr.edu/

**Coverage:** Global, 19,000+ stations

**Pros:**
- High-quality processed data
- Easy-to-use web interface
- Daily tropospheric delay estimates
- Well-documented format

**How to Download:**

1. Go to http://geodesy.unr.edu/
2. Click **"Data"** → **"Data Archive"**
3. **Select region:** Use map or search by station name
   - For Poland (bogo_pl example): Look for stations like WROC, KATO, LODZ
4. **Download files:**
   - Select **.trop** or **.ztd** files (tropospheric delays)
   - Choose date range (e.g., 2023-01-01 to 2023-12-31)
5. **Convert to CSV:**
   ```python
   # Example conversion script
   import pandas as pd

   # Parse Nevada format (whitespace-separated)
   df = pd.read_csv('station.trop', delim_whitespace=True,
                    names=['year', 'doy', 'hour', 'ztd', 'sigma'],
                    comment='%')

   # Convert to datetime
   df['datetime'] = pd.to_datetime(df['year']*1000 + df['doy'], format='%Y%j')

   # Add station metadata
   df['station_id'] = 'WROC'
   df['lat'] = 51.113
   df['lon'] = 17.062

   # Save in standard format
   df[['station_id', 'lat', 'lon', 'ztd', 'datetime']].to_csv('gnss_data.csv', index=False)
   ```

---

### 3. EUREF Permanent GNSS Network (Manual)

**Website:** http://www.epncb.oma.be/

**Coverage:** Europe, 300+ stations

**Pros:**
- Dense network in Europe
- Official EUREF products
- Quality-controlled data

**How to Download:**

1. Go to http://www.epncb.oma.be/
2. Select **"Products"** → **"Troposphere"**
3. Choose **date range** and **stations**
4. Download **ZTD files** (SINEX or ASCII format)
5. Parse and convert to CSV format

**Example Stations in Poland:**
- WROC (Wrocław): 51.113°N, 17.062°E
- BOGO (Borowiec): 52.277°N, 17.073°E
- BYDG (Bydgoszcz): 53.124°N, 18.008°E
- KATO (Katowice): 50.265°N, 19.024°E

---

### 4. International GNSS Service (IGS)

**Website:** https://igs.org/

**Coverage:** Global, 500+ core stations

**Pros:**
- Highest quality data
- Long-term stability
- International standard

**How to Download:**

1. Go to https://igs.org/
2. Navigate to **"Data & Products"** → **"Troposphere"**
3. Select **IGS Analysis Center** (e.g., JPL, CODE, GFZ)
4. Download **troposphere products** (SINEX or ZPD format)
5. Use IGS tools or custom parser to extract ZTD

**Data Centers:**
- ftp://igs.ign.fr/pub/igs/products/troposphere/
- ftp://cddis.gsfc.nasa.gov/pub/gps/products/troposphere/

---

### 5. EPOS GNSS Data Portal (Manual)

**Website:** https://gnssdata-epos.oca.eu/

**Coverage:** European area

**Pros:**
- Modern web interface
- Direct browser access
- Multiple data types

**How to Use:**

1. Go to https://gnssdata-epos.oca.eu/GlassFramework/
2. Use web interface to browse stations
3. Select region and time period
4. Download processed products
5. Convert to standard CSV format

**Note:** API is available but endpoint structure may change. Manual download is more reliable.

---

### 6. Local/Regional Networks

Many countries have their own GNSS networks:

**Poland:**
- ASG-EUPOS: http://www.asgeupos.pl/
- Polish national geodetic network

**Other Countries:**
- Check with national geodetic survey
- University research networks
- Regional geodetic services

---

## Standard Data Format

All data should be converted to this CSV format:

```csv
station_id,lat,lon,ztd,datetime
WROC,51.113,17.062,2.345,2023-06-07T04:44:45
BOGO,52.277,17.073,2.389,2023-06-07T05:00:00
KATO,50.265,19.024,2.312,2023-06-07T04:30:00
```

### Column Definitions:

- **station_id**: Unique 4-character station code (e.g., WROC, BOGO)
- **lat**: Latitude in decimal degrees (WGS84)
- **lon**: Longitude in decimal degrees (WGS84)
- **ztd**: Zenith Tropospheric Delay in meters
- **datetime**: ISO 8601 format (YYYY-MM-DDTHH:MM:SS)

---

## Loading Data

### Method 1: From CSV File

```bash
python scripts/download_gnss_data.py \
    --config data/configs/gnss/2023/bogo_pl.json \
    --method file \
    --file path/to/gnss_data.csv \
    --format csv
```

### Method 2: From Nevada Format

```bash
python scripts/download_gnss_data.py \
    --config data/configs/gnss/2023/bogo_pl.json \
    --method file \
    --file path/to/station.trop \
    --format nevada
```

### Method 3: From Python

```python
from libs.internal.gnss import GnssDownloader

downloader = GnssDownloader('data/configs/gnss/2023/bogo_pl.json')

# Load from file
gnss_data = downloader.load_gnss_from_file(
    'path/to/gnss_data.csv',
    file_format='csv'
)

# Check data
print(f"Loaded {len(gnss_data)} measurements")
print(f"Stations: {gnss_data['station_id'].unique()}")
print(f"Date range: {gnss_data['datetime'].min()} to {gnss_data['datetime'].max()}")
```

---

## Data Quality Considerations

### Temporal Resolution

- **Daily averages:** Sufficient for most InSAR validation
- **Hourly data:** Better for matching SAR acquisition times
- **30-minute or better:** Ideal for high-temporal resolution studies

### Spatial Coverage

For your study area, aim for:
- **Minimum:** 3-5 stations within 100 km
- **Good:** 10+ stations within 50 km
- **Ideal:** Dense network with <20 km spacing

### Data Quality Flags

When downloading:
- Check for data gaps
- Verify coordinates match station locations
- Look for quality flags or sigma values
- Remove outliers (ZTD < 1.5 m or > 3.5 m is suspicious)

---

## Example: Downloading for Poland

For the `bogo_pl` study area (around Borowiec, Poland):

### Recommended Stations:

| Station | Name | Lat | Lon | Network | Distance from Bogo |
|---------|------|-----|-----|---------|-------------------|
| BOGO | Borowiec | 52.277 | 17.073 | EUREF | 0 km |
| WROC | Wrocław | 51.113 | 17.062 | EUREF | ~130 km |
| POZN | Poznań | 52.384 | 16.926 | ASG-EUPOS | ~15 km |
| LAMA | Lamkówko | 53.892 | 20.667 | EUREF | ~250 km |

### Download Steps:

1. **Nevada Geodetic Lab:**
   - Search for "WROC" and "BOGO"
   - Download .trop files for 2023
   - Repeat for other stations

2. **EUREF:**
   - Access troposphere products
   - Download SINEX files for all EPN stations
   - Filter for Polish stations

3. **Convert to CSV:**
   ```bash
   # Use provided conversion tools or custom script
   python scripts/convert_gnss_to_csv.py --input *.trop --output gnss_poland_2023.csv
   ```

4. **Load into pipeline:**
   ```bash
   python scripts/download_gnss_data.py \
       --config data/configs/gnss/2023/bogo_pl.json \
       --method file \
       --file gnss_poland_2023.csv
   ```

---

## Troubleshooting

### Problem: No data available for my dates

**Solution:**
- Check data latency (usually 1-7 days after collection)
- Try different analysis centers
- Use synthetic data for testing

### Problem: Station coordinates don't match

**Solution:**
- Verify station codes (4-character IGS standard)
- Check for station moves/equipment changes
- Use official station logs: https://igs.org/network/

### Problem: ZTD values seem wrong

**Solution:**
- Typical ZTD: 2.0-2.7 meters (varies by season/location)
- Higher in summer (more water vapor)
- Check units (should be meters, not millimeters)
- Verify datetime parsing is correct

---

## API Status (2026-05-04)

**Copernicus Climate Data Store API:** ✅
- Endpoint: `https://cds.climate.copernicus.eu/api`
- Status: **Operational and recommended**
- Documentation: https://cds.climate.copernicus.eu/datasets/insitu-observations-gnss
- Python package: `cdsapi` (install with `pip install cdsapi`)

**Features:**
- Automated downloads via Python API
- Global IGS and European EPN networks
- Reprocessed data (consistent quality)
- Near real-time daily products
- Hourly ZTD measurements
- CSV and NetCDF formats

**Requirements:**
- Free CDS account registration
- API credentials in `~/.cdsapirc`
- `cdsapi` Python package

---

## Alternative: Synthetic Data

For **testing and development**, synthetic data is sufficient:

```bash
python scripts/download_gnss_data.py \
    --config data/configs/gnss/2023/bogo_pl.json \
    --method synthetic
```

**Synthetic data includes:**
- 5-10 stations randomly placed in AOI
- Realistic ZTD values (2.0-2.7 m)
- Seasonal variation (summer/winter)
- Daily time series for full year
- Random noise (~2 cm)

**Use for:**
- Pipeline testing
- Algorithm development
- Training demonstrations

**Do NOT use for:**
- Scientific validation
- Publication results
- Accuracy assessment

---

## Summary

**Best Practice Workflow:**

1. **For Testing:** Use synthetic data
   ```bash
   python scripts/download_gnss_data.py \
       --config data/configs/gnss/2023/bogo_pl.json \
       --method synthetic
   ```

2. **For Production (Automated):** ⭐ **Recommended**
   - Register for free CDS account
   - Configure API credentials
   - Use automated download:
   ```bash
   python scripts/download_gnss_data.py \
       --config data/configs/gnss/2023/bogo_pl.json \
       --method api
   ```

3. **For Production (Manual):**
   - Manually download from Nevada Geodetic Lab or EUREF
   - Convert to standard CSV format
   - Load into pipeline with `--method file`

4. **For Validation:**
   - Use multiple stations (5-10 minimum)
   - Ensure good spatial coverage
   - Check data quality
   - Compare with GACOS corrections

---

**Last Updated:** 2026-05-04
**Recommended Source:** Copernicus Climate Data Store (https://cds.climate.copernicus.eu/datasets/insitu-observations-gnss)
**API Access:** Automated via `cdsapi` Python package
**Format:** Standard CSV with columns: station_id, lat, lon, ztd, datetime
