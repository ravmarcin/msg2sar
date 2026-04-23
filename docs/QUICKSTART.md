# MSG2SAR Atmospheric Correction - Quick Start Guide

## What's Been Implemented

This implementation adds ML-based atmospheric correction capabilities to msg2sar. The following components are **READY TO USE**:

### ✅ Phase 1: Foundation (Complete)

1. **GNSS System** - Download and process GNSS tropospheric delay data
   - `utils/internal/gnss/` module
   - EPOS API integration
   - Spatial/temporal interpolation
   - Config: `data/configs/gnss/2023/bogo_pl.json`

2. **GACOS System** - Apply GACOS atmospheric corrections
   - `utils/internal/gacos/` module
   - Download GACOS .ztd files
   - Resample to SAR grid
   - Apply corrections to interferograms
   - Config: `data/configs/gacos/2023/bogo_pl.json`

### ✅ Phase 2: SEVIRI Processing (Complete)

3. **SEVIRI Temporal Downsampling** - Match SEVIRI to SAR timing
   - `utils/internal/msg/seviri_temporal.py`
   - Multi-channel temporal interpolation
   - Batch processing with configurable tiles
   - Output: `seviri_temporal_stack.nc`

4. **SEVIRI-SAR Geometry** - Convert vertical to slant geometry
   - `utils/internal/msg/seviri_sar_geometry.py`
   - LOS geometry computation
   - Atmospheric integration along ray path
   - Output: `seviri_los_geometry.nc`

### ✅ Phase 3: ML Data Preparation (Partial)

5. **ML Data Config and Loader** - PyTorch data pipeline
   - `utils/internal/ml/data_config.py`
   - `utils/internal/ml/data_loader.py`
   - 14-channel input (SEVIRI + coherence)
   - 1-channel output (GACOS-corrected phase)
   - Data augmentation and normalization

---

## Quick Usage Examples

### 1. Download GNSS Data

```python
from utils.internal.gnss import GnssDownloader
from datetime import datetime

# Initialize downloader
downloader = GnssDownloader('data/configs/gnss/2023/bogo_pl.json')

# Download for date range
gnss_data = downloader.download_gnss_stations(
    aoi=downloader.config.aoi,
    date_range=(datetime(2023, 1, 1), datetime(2023, 12, 31))
)

# Compute ZTD at reference point
ztd = downloader.compute_reference_ztd(
    sar_timestamp=datetime(2023, 6, 7, 4, 44, 45)
)
```

### 2. Apply GACOS Correction

```python
from utils.internal.gacos import GacosProcessor
from datetime import datetime
import xarray as xr

# Initialize processor
processor = GacosProcessor('data/configs/gacos/2023/bogo_pl.json')

# Load interferogram
intf = xr.open_dataarray('path/to/interferogram.nc')
inc_angle = xr.open_dataarray('path/to/incidence_angle.nc')

# Apply correction
corrected = processor.process_interferogram(
    interferogram=intf,
    master_date=datetime(2023, 6, 7),
    slave_date=datetime(2023, 7, 1),
    los_inc_angle=inc_angle,
    output_filename='corrected_20230607_20230701.nc'
)
```

### 3. Process SEVIRI Temporal Stack

```python
from utils.internal.msg.seviri_temporal import SeviriTemporalProcessor
from utils.internal.msg.msg_config import MsgConfig
from utils.internal.msg.pymsg.stack_base import MsgStackBase
from datetime import datetime

# Initialize
config = MsgConfig('data/configs/msg/seviri/2023/bogo_pl.json')
processor = SeviriTemporalProcessor(config)

# Create MSG stack
stack = MsgStackBase(
    data_dir=config.download_dir,
    work_dir=config.process_dir,
    geojson_path=config.aoi_path,
    projection=config.msg_processing['raster']['projection']
)

# Process temporal stack (assumes seviri_channels already loaded)
temporal_stack = processor.process_temporal_stack(
    stack_base=stack,
    sar_timestamp=datetime(2023, 6, 7, 4, 44, 45),
    seviri_data_dict=seviri_channels,
    n_images=10
)

# Save
processor.save_temporal_stack(temporal_stack)
```

### 4. Compute SAR Geometry

```python
from utils.internal.msg.seviri_sar_geometry import SeviriSarGeometryConverter
from utils.internal.msg.msg_config import MsgConfig
import numpy as np

# Initialize
config = MsgConfig('data/configs/msg/seviri/2023/bogo_pl.json')
converter = SeviriSarGeometryConverter(config)

# Compute LOS geometry
sar_lats = np.linspace(50.0, 51.0, 100)
sar_lons = np.linspace(20.0, 21.0, 100)

los_geometry = converter.compute_los_mapping(
    eof_path='path/to/orbit_file.EOF',
    acquisition_time='2023-06-07T04:44:45',
    sar_lats=sar_lats,
    sar_lons=sar_lons
)

# Save
converter.save_los_geometry(los_geometry)
```

### 5. Prepare ML Data

```python
from utils.internal.ml import MLDataConfig, AtmosphericCorrectionDataset
from utils.internal.ml.data_loader import get_data_loaders

# Initialize config
config = MLDataConfig('data/configs/ml/atmospheric_correction/bogo_pl_master.json')

# Create data loaders
train_loader, val_loader = get_data_loaders(
    config,
    seviri_path='path/to/seviri_temporal_stack.nc',
    coherence_path='path/to/coherence.nc',
    phase_path='path/to/gacos_corrected_phase.nc'
)

# Iterate
for inputs, targets in train_loader:
    # inputs: [batch, 14, 256, 256]
    # targets: [batch, 1, 128, 128]
    pass
```

---

## Next Steps to Complete Implementation

### Immediate (< 1 week)

1. **Implement UNet Model** (`utils/internal/ml/models/unet.py`)
   - 5-level encoder-decoder architecture
   - Skip connections
   - Center cropping (256→128)

2. **Create Training Loop** (`utils/internal/ml/trainer.py`)
   - Training/validation loop
   - TensorBoard logging
   - Checkpoint management
   - Early stopping

3. **Create Training Script** (`scripts/train_atmospheric_correction.py`)
   - Load config and data
   - Initialize model
   - Run training
   - Save best model

### Short-term (1-2 weeks)

4. **Testing Infrastructure**
   - Unit tests for all modules
   - Integration tests
   - Use pytest framework
   - Target >90% coverage

5. **End-to-End Pipeline**
   - Data preparation script
   - Training script
   - Inference script
   - Validation script

### Medium-term (2-4 weeks)

6. **Documentation**
   - API documentation
   - Usage examples
   - Troubleshooting guide
   - Performance tuning guide

7. **Validation**
   - Test on bogo_pl dataset
   - Compare with GACOS baseline
   - Validate against GNSS
   - Performance metrics

---

## File Structure

```
msg2sar/
├── utils/internal/
│   ├── gnss/                    ✅ Complete
│   │   ├── __init__.py
│   │   ├── gnss_config.py
│   │   ├── gnss_downloader.py
│   │   └── gnss_processor.py
│   ├── gacos/                   ✅ Complete
│   │   ├── __init__.py
│   │   ├── gacos_config.py
│   │   └── gacos_processor.py
│   ├── msg/
│   │   ├── seviri_temporal.py       ✅ Complete
│   │   └── seviri_sar_geometry.py   ✅ Complete
│   └── ml/                      🚧 Partial
│       ├── __init__.py              ✅
│       ├── data_config.py           ✅
│       ├── data_loader.py           ✅
│       ├── trainer.py               ⏳ TODO
│       └── models/
│           ├── __init__.py          ⏳ TODO
│           └── unet.py              ⏳ TODO
│
├── data/configs/
│   ├── gnss/2023/bogo_pl.json              ✅
│   ├── gacos/2023/bogo_pl.json             ✅
│   ├── msg/seviri/2023/bogo_pl.json        ✅ (extended)
│   └── ml/atmospheric_correction/
│       └── bogo_pl_master.json             ✅
│
├── scripts/                     ⏳ TODO
│   ├── prepare_ml_data.py
│   ├── train_atmospheric_correction.py
│   └── inference_atmospheric_correction.py
│
├── tests/unittests/             ⏳ TODO
│   └── utils/internal/
│       ├── gnss/
│       ├── gacos/
│       ├── msg/
│       └── ml/
│
├── IMPLEMENTATION_STATUS.md      ✅ Full tracking doc
└── QUICKSTART.md                 ✅ This file
```

---

## Installation

### Dependencies Added

Already added to `env_build/pip_requirements.txt`:
```
torch==2.1.0
torchvision==0.16.0
tensorboard==2.15.0
pytest==7.4.3
pytest-cov==4.1.0
pytest-mock==3.12.0
albumentations==1.3.1
scipy==1.11.4
```

### Install

```bash
# Update conda environment
conda env update -f env_build/environment.yml

# Install pip dependencies
pip install -r env_build/pip_requirements.txt
```

---

## Testing

### Manual Testing

Test each component individually:

```python
# Test GNSS
python -c "from utils.internal.gnss import GnssConfig; c = GnssConfig('data/configs/gnss/2023/bogo_pl.json'); print(c.job_name)"

# Test GACOS
python -c "from utils.internal.gacos import GacosConfig; c = GacosConfig('data/configs/gacos/2023/bogo_pl.json'); print(c.job_name)"

# Test ML Config
python -c "from utils.internal.ml import MLDataConfig; c = MLDataConfig('data/configs/ml/atmospheric_correction/bogo_pl_master.json'); print(c.job_name)"
```

### Unit Tests (TODO)

```bash
# Run all tests
pytest tests/unittests/ -v

# With coverage
pytest tests/unittests/ --cov=utils/internal --cov-report=html
```

---

## Configuration

The master config file integrates all components:
- **Location:** `data/configs/ml/atmospheric_correction/bogo_pl_master.json`
- **Contains:** References to SAR, MSG, GNSS, GACOS configs
- **Includes:** ML training hyperparameters, model architecture, pipeline settings

---

## Support

For implementation questions:
1. Check `IMPLEMENTATION_STATUS.md` for detailed progress
2. Review code comments in each module
3. Reference the original implementation plan

---

## Progress Summary

| Phase | Component | Status | Progress |
|-------|-----------|--------|----------|
| 1 | GNSS System | ✅ Complete | 100% |
| 1 | GACOS System | ✅ Complete | 100% |
| 2 | SEVIRI Temporal | ✅ Complete | 100% |
| 2 | SEVIRI Geometry | ✅ Complete | 100% |
| 3 | ML Data Prep | 🚧 Partial | 75% |
| 3 | UNet Model | ⏳ TODO | 0% |
| 3 | Testing | ⏳ TODO | 0% |
| 4 | Integration | ⏳ TODO | 0% |

**Overall Progress: ~60% Complete**

---

**Last Updated:** 2026-04-23
