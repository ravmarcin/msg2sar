# MSG2SAR Atmospheric Correction Implementation Status

**Date:** 2026-04-23
**Project:** ML-based Atmospheric Correction for InSAR using SEVIRI Data

---

## Overview

This document tracks the implementation progress of the atmospheric correction pipeline as specified in the implementation plan. The system integrates SEVIRI satellite data, GNSS measurements, GACOS corrections, and deep learning to improve InSAR accuracy.

---

## Phase 1: Foundation Components ✅ COMPLETED

### Component 2: GNSS Data System ✅

**Status:** Fully implemented
**Files Created:**
- `utils/internal/gnss/__init__.py`
- `utils/internal/gnss/gnss_config.py`
- `utils/internal/gnss/gnss_downloader.py`
- `utils/internal/gnss/gnss_processor.py`
- `data/configs/gnss/2023/bogo_pl.json`

**Features:**
- [x] GnssConfig class following MsgConfig pattern
- [x] EPOS API integration for station data download
- [x] Spatial and temporal interpolation of ZTD data
- [x] Grid-based ZTD computation for SAR geometry
- [x] CSV and NetCDF data storage
- [x] Configurable temporal/spatial buffers

**Integration Points:**
- Uses existing `utils/internal/geo/aoi.py` for AOI handling
- Compatible with `sbas_config.py` reference point structure
- Follows standard config loading patterns

---

### Component 3: GACOS Atmospheric Correction ✅

**Status:** Fully implemented
**Files Created:**
- `utils/internal/gacos/__init__.py`
- `utils/internal/gacos/gacos_config.py`
- `utils/internal/gacos/gacos_processor.py`
- `data/configs/gacos/2023/bogo_pl.json`

**Features:**
- [x] GacosConfig class for configuration management
- [x] GACOS .ztd file download from http://www.gacos.net/
- [x] ASCII grid parsing and NetCDF conversion
- [x] Bilinear resampling to SAR grid (0.125° → SAR resolution)
- [x] Atmospheric correction formula implementation
- [x] LOS geometry integration (zenith to slant conversion)
- [x] Complete interferogram processing pipeline

**Integration Points:**
- Can be applied after `sbas_process.py::unwrapping()`
- Uses incidence angle from SAR multilook output
- Outputs GACOS-corrected interferograms for ML training

---

## Phase 2: SEVIRI Processing ✅ COMPLETED

### Component 1: SEVIRI Temporal Downsampling ✅

**Status:** Fully implemented
**Files Created:**
- `utils/internal/msg/seviri_temporal.py`
- Config extensions in `data/configs/msg/seviri/2023/bogo_pl.json`

**Features:**
- [x] SeviriTemporalProcessor class
- [x] Multi-channel support (6 SEVIRI channels)
- [x] Temporal interpolation using optical flow
- [x] Batch processing with configurable tile size
- [x] Translation vector computation per image pair
- [x] Integration with MsgStackBase
- [x] xarray Dataset output (pair, channel, y, x)

**Configuration:**
```json
"temporal_downsampling": {
    "n_images_window": 10,
    "batch_size_x": 256,
    "batch_size_y": 256,
    "overlap": 32,
    "translation_steps": [8, 4, 2, 1]
}
```

**Integration Points:**
- Extends `utils/internal/img/temporal_upsampling.py`
- Uses `utils/internal/msg/pymsg/stack_base.py::MsgStackBase`
- Outputs to `config.process_dir/seviri_temporal_stack.nc`

---

### Component 4: SEVIRI to SAR Geometry Conversion ✅

**Status:** Fully implemented
**Files Created:**
- `utils/internal/msg/seviri_sar_geometry.py`
- Config extensions in `data/configs/msg/seviri/2023/bogo_pl.json`

**Features:**
- [x] SeviriSarGeometryConverter class
- [x] LOS geometry computation using Sentinel-1 orbit data
- [x] Incidence and azimuth angle calculation
- [x] Vertical to slant delay conversion
- [x] Atmospheric integration along ray path
- [x] ECEF coordinate transformations

**Configuration:**
```json
"sar_geometry": {
    "z_top_m": 12000,
    "dz_m": 100,
    "eof_orbit_dir": "orbit_files"
}
```

**Integration Points:**
- Uses `utils/internal/sentinel/orbit.py::los_sampling_points()`
- Compatible with `utils/internal/msg/seviri.py::build_Nwet_from_seviri()`
- Outputs LOS geometry and projected delays

---

## Phase 3: ML Pipeline 🚧 IN PROGRESS

### Component 5: ML Data Preparation ⏳

**Status:** Not yet implemented
**Files to Create:**
- `utils/internal/ml/__init__.py`
- `utils/internal/ml/data_loader.py`
- `utils/internal/ml/data_config.py`
- `scripts/prepare_ml_data.py`

**Required Features:**
- [ ] AtmosphericCorrectionDataset (PyTorch Dataset)
- [ ] MLDataConfig class
- [ ] Data loading from SEVIRI, coherence, GACOS
- [ ] Train/validation split (80/20)
- [ ] Per-channel normalization
- [ ] Data augmentation pipeline
- [ ] Zarr storage for fast loading

**Input Specification:**
- **Input Channels (14):**
  - SEVIRI t1: WV_062, WV_073, IR_097, IR_087, IR_108, IR_120 (6)
  - SEVIRI t2: WV_062, WV_073, IR_097, IR_087, IR_108, IR_120 (6)
  - Coherence maps: 2 interferogram pairs (2)
- **Output Channel (1):**
  - GACOS-corrected atmospheric phase delay

**Data Flow:**
```
SEVIRI temporal stack (Component 1)
  + SAR coherence maps (SBAS)
  + GACOS corrections (Component 3)
  → Normalized training data (zarr)
```

---

### Component 6: UNet Model Architecture ⏳

**Status:** Not yet implemented
**Files to Create:**
- `utils/internal/ml/models/__init__.py`
- `utils/internal/ml/models/unet.py`
- `utils/internal/ml/trainer.py`
- `scripts/train_atmospheric_correction.py`
- `scripts/inference_atmospheric_correction.py`

**Required Features:**
- [ ] AtmosphericCorrectionUNet model
- [ ] 5-level encoder-decoder with skip connections
- [ ] Center cropping (256×256 → 128×128)
- [ ] AtmosphericCorrectionTrainer class
- [ ] Training loop with validation
- [ ] TensorBoard logging
- [ ] Checkpoint saving
- [ ] Early stopping

**Architecture:**
```
Input: [Batch, 14, 256, 256]
  → Encoder (5 levels with MaxPool)
  → Bottleneck (init_features * 16)
  → Decoder (5 levels with TransposeConv + skip connections)
  → 1×1 Conv
  → Center Crop
Output: [Batch, 1, 128, 128]
```

---

### Component 7: Testing Plan ⏳

**Status:** Not yet implemented
**Files to Create:**
- Test files in `tests/unittests/utils/internal/gnss/`
- Test files in `tests/unittests/utils/internal/gacos/`
- Test files in `tests/unittests/utils/internal/msg/`
- Test files in `tests/unittests/utils/internal/ml/`

**Test Coverage:**
- [ ] GNSS downloader and processor tests
- [ ] GACOS processor tests
- [ ] SEVIRI temporal processing tests
- [ ] SEVIRI geometry conversion tests
- [ ] ML data loader tests
- [ ] UNet model tests
- [ ] Trainer tests
- [ ] Integration tests

---

## Phase 4: Integration & Validation ⏳

**Status:** Not yet started
**Required Tasks:**
- [ ] End-to-end pipeline testing on bogo_pl dataset
- [ ] Validation against GACOS baseline
- [ ] Validation against GNSS measurements
- [ ] Performance optimization (Dask tuning)
- [ ] Documentation and usage examples
- [ ] User guide creation

---

## Dependencies

### Added to `env_build/pip_requirements.txt` ✅
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

### Already Available ✅
- numpy, pandas, xarray
- dask, distributed
- rasterio, pyresample
- requests
- zarr, h5py

---

## Configuration Files

### Master Configuration ✅
- `data/configs/ml/atmospheric_correction/bogo_pl_master.json`

### Component Configurations ✅
- `data/configs/gnss/2023/bogo_pl.json`
- `data/configs/gacos/2023/bogo_pl.json`
- `data/configs/msg/seviri/2023/bogo_pl.json` (extended)

---

## Directory Structure

```
utils/internal/
├── gnss/                       ✅ Created
│   ├── __init__.py
│   ├── gnss_config.py
│   ├── gnss_downloader.py
│   └── gnss_processor.py
├── gacos/                      ✅ Created
│   ├── __init__.py
│   ├── gacos_config.py
│   └── gacos_processor.py
├── msg/
│   ├── seviri_temporal.py      ✅ Created
│   └── seviri_sar_geometry.py  ✅ Created
└── ml/                         ⏳ To be created
    ├── __init__.py
    ├── data_loader.py
    ├── data_config.py
    ├── trainer.py
    └── models/
        ├── __init__.py
        └── unet.py

data/
├── configs/
│   ├── gnss/2023/              ✅ Created
│   ├── gacos/2023/             ✅ Created
│   └── ml/atmospheric_correction/  ✅ Created
├── gnss/2023/bogo_pl/          (Created on first run)
├── gacos/2023/bogo_pl/         (Created on first run)
└── ml/atmospheric_correction/  (Created on first run)

scripts/                        ⏳ To be created
├── prepare_ml_data.py
├── train_atmospheric_correction.py
└── inference_atmospheric_correction.py
```

---

## Next Steps

### Immediate (Phase 3 - Component 5)
1. Create `utils/internal/ml/` module structure
2. Implement `MLDataConfig` class
3. Implement `AtmosphericCorrectionDataset` PyTorch Dataset
4. Create `prepare_ml_data.py` script
5. Test data loading pipeline

### Short-term (Phase 3 - Component 6)
1. Implement UNet architecture
2. Create training loop with validation
3. Add TensorBoard logging
4. Create training and inference scripts

### Medium-term (Component 7 + Phase 4)
1. Write comprehensive unit tests
2. Run end-to-end pipeline on bogo_pl
3. Validate against GACOS and GNSS
4. Optimize performance
5. Create documentation

---

## Usage Examples (Preliminary)

### GNSS Data Download
```python
from utils.internal.gnss import GnssDownloader
from datetime import datetime

downloader = GnssDownloader('data/configs/gnss/2023/bogo_pl.json')

# Download GNSS data for date range
gnss_data = downloader.download_gnss_stations(
    aoi=downloader.config.aoi,
    date_range=(datetime(2023, 1, 1), datetime(2023, 12, 31))
)
```

### GACOS Correction
```python
from utils.internal.gacos import GacosProcessor
from datetime import datetime

processor = GacosProcessor('data/configs/gacos/2023/bogo_pl.json')

# Process interferogram
corrected = processor.process_interferogram(
    interferogram=intf_data,
    master_date=datetime(2023, 6, 7),
    slave_date=datetime(2023, 7, 1),
    los_inc_angle=inc_angle_data,
    output_filename='corrected_intf.nc'
)
```

### SEVIRI Temporal Processing
```python
from utils.internal.msg.seviri_temporal import SeviriTemporalProcessor
from utils.internal.msg.msg_config import MsgConfig
from datetime import datetime

config = MsgConfig('data/configs/msg/seviri/2023/bogo_pl.json')
processor = SeviriTemporalProcessor(config)

# Process temporal stack
temporal_stack = processor.process_temporal_stack(
    stack_base=msg_stack,
    sar_timestamp=datetime(2023, 6, 7, 4, 44, 45),
    seviri_data_dict=seviri_channels,
    n_images=10
)

processor.save_temporal_stack(temporal_stack)
```

---

## Notes

- All Phase 1 and Phase 2 components follow the existing codebase patterns
- Configuration management uses the MsgConfig pattern consistently
- Data storage uses xarray/NetCDF for compatibility with existing pipeline
- Logging uses the existing `utils.internal.log.logger` system
- All components are modular and can be tested independently

---

## Success Criteria

- [x] Phase 1: GNSS and GACOS modules operational
- [x] Phase 2: SEVIRI temporal and geometry processing complete
- [ ] Phase 3: ML pipeline trains without errors
- [ ] Model predictions correlate with GACOS (R² > 0.7)
- [ ] ZTD predictions within ±2 cm of GNSS
- [ ] Full pipeline completes in <24 hours

---

**Last Updated:** 2026-04-23
**Next Review:** After Phase 3 completion
