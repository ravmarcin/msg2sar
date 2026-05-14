# MSG2SAR Atmospheric manual - Implementation Plan & Status

**Last Updated:** 2026-05-14

---

## Legend

Each component is tracked with 4 checkmarks:

| Step | Meaning |
|------|---------|
| **ai** | Code generated / written |
| **test** | Unit tests written and passing |
| **manual** | Code reviewed, corrected after testing |
| **integration** | Integrated into the end-to-end pipeline |

---

## Phase 1: Foundation Components

### Component 1a: GNSS ZTD Data Downloader

**Files:** `libs/internal/gnss/` (gnss_config.py, gnss_downloader.py, gnss_processor.py)
**Config:** `data/configs/gnss/2022/bogo_pl.json`
**Script:** `scripts/download_gnss_data.py`

- [x] ai
- [x] test
- [x] manual
- [ ] integration

**Notes:** Copernicus CDS API for ZTD and TCWV download (incl. ERA5 water vapour column). EPOS API integration. Spatial/temporal interpolation. Synthetic data generation for testing. Networks: IGS (global), EPN (European).

---

### Component 1b: GNSS Position Data Downloader

**Files:** to be created
**Config:** to be extended in `data/configs/gnss/`

- [ ] ai
- [ ] test
- [ ] manual
- [ ] integration

**Notes:** Download GNSS station position/coordinate data (not just ZTD). Required for precise station metadata, receiver/antenna info, and coordinate time series. Currently station coordinates are only obtained as a byproduct of CDS ZTD downloads.

---

### Component 2: InSAR/SBAS Processor

**Files:** `libs/internal/sbas/` (sbas_process.py, sbas_spec.py, sbas_downloader.py, sbas_config.py, utils.py, local_setup.py)
**Config:** `data/configs/sar/sbas/desc/2023/bogo_pl.json`
**Scripts:** `scripts/sbas_download.py`, `scripts/sbas_preprocess.py`, `scripts/sbas_align.py`, `scripts/sbas_geocode.py`

- [x] ai
- [x] test
- [x] manual
- [ ] integration

**Notes:** Complete SBAS workflow: Sentinel-1 burst download (ASF API), orbit download, reframing, alignment, geocoding, multilook, interferogram generation (Goldstein filtering), SBAS pair selection, SNAPHU phase unwrapping, topographic trend removal, LOS displacement. Uses pygmtsar + Dask.

---

### Component 3: SEVIRI Downloader

**Files:** `libs/internal/msg/download.py`, `libs/internal/msg/msg_downloader.py`
**Config:** `data/configs/msg/seviri/2023/bogo_pl.json`

- [x] ai
- [x] test
- [x] manual
- [ ] integration

**Notes:** EUMETSAT EUMDAC API for MSG-SEVIRI L1.5 HRIT/NetCDF and MTG-IRS L1/L2 download. Spatial filtering via GeoJSON polygons. Collections: SEVIRI (EO:EUM:DAT:0665), HRSEVIRI (L1.5), WV_002_013 (L2 water vapour/TPW).

---

### Component 4: ERA5 Vertical manual for SEVIRI

**Files:** to be created
**Config:** to be defined

- [ ] ai
- [ ] test
- [ ] manual
- [ ] integration

**Notes:** ERA5 reanalysis-based vertical manual for SEVIRI imagery. Use ERA5 atmospheric profiles (temperature, humidity, pressure) to correct SEVIRI-derived atmospheric delays. CDS API already available in GNSS downloader can be extended for ERA5 model-level/pressure-level data.

---

### Component 5: GACOS Atmospheric manual

**Files:** `libs/internal/gacos/` (gacos_config.py, gacos_processor.py)
**Config:** `data/configs/gacos/2023/bogo_pl.json`

- [ ] ai
- [ ] test
- [ ] manual
- [ ] integration

**Notes:** GACOS .ztd file download, ASCII grid parsing, bilinear resampling to SAR grid, zenith-to-slant conversion, interferogram manual pipeline.

---

## Phase 2: SEVIRI Processing

### Component 6: SEVIRI Temporal Downsampling

**Files:** `libs/internal/msg/seviri_temporal.py`
**Config:** `data/configs/msg/seviri/2023/bogo_pl.json` (extended)

- [x] ai
- [ ] test
- [ ] manual
- [ ] integration

**Notes:** Multi-channel support (6 SEVIRI channels), optical flow temporal interpolation, batch processing with configurable tile size, xarray Dataset output.

---

### Component 7: SEVIRI to SAR Geometry Conversion

**Files:** `libs/internal/msg/seviri_sar_geometry.py`
**Config:** `data/configs/msg/seviri/2023/bogo_pl.json` (extended)

- [x] ai
- [ ] test
- [ ] manual
- [ ] integration

**Notes:** LOS geometry computation using Sentinel-1 orbit data, incidence/azimuth angle calculation, vertical-to-slant delay conversion, ECEF coordinate transformations.

---

## Phase 3: ML Pipeline

### Component 8: ML Data Preparation

**Files:** `libs/internal/ml/` (data_config.py, data_loader.py)
**Config:** `data/configs/ml/atmospheric_manual/bogo_pl_master.json`
**Script:** `scripts/prepare_ml_data.py`

- [x] ai
- [ ] test
- [ ] manual
- [ ] integration

**Notes:** PyTorch Dataset (14 input channels: SEVIRI t1+t2 + coherence; 1 output channel: GACOS-corrected phase), per-channel normalization, data augmentation (Albumentations), train/val split (80/20).

---

### Component 9: UNet Model Architecture

**Files:** `libs/internal/ml/models/` (unet.py, __init__.py), `libs/internal/ml/trainer.py`
**Scripts:** `scripts/train_atmospheric_manual.py`, `scripts/inference_atmospheric_manual.py`, `scripts/test_model_mps.py`

- [x] ai
- [ ] test
- [ ] manual
- [ ] integration

**Notes:** ~31M parameter UNet, 5-level encoder-decoder with skip connections, center cropping (256x256 -> 128x128). Trainer with TensorBoard logging, checkpoint management, early stopping, LR scheduling. Apple Silicon (MPS) GPU acceleration with automatic device detection.

---

### Component 10: Unit & Integration Tests

**Files to create:** `tests/unittests/utils/internal/{gnss,gacos,msg,ml}/`
**Existing:** only `tests/unittests/utils/internal/img/test_temporal_upsampling.py` (pre-existing, unrelated)

- [ ] ai
- [ ] test
- [ ] manual
- [ ] integration

**Notes:** No test files created yet for any of the new modules. Test coverage needed for: GNSS downloader/processor, GACOS processor, SEVIRI temporal processing, SEVIRI geometry conversion, ML data loader, UNet model, trainer.

---

## Phase 4: End-to-End Validation

### Component 11: Pipeline Integration & Validation

- [ ] ai
- [ ] test
- [ ] manual
- [ ] integration

**Tasks:**
- [ ] End-to-end pipeline testing on bogo_pl dataset
- [ ] Validation against GACOS baseline (target: R2 > 0.7)
- [ ] Validation against GNSS measurements (target: ZTD within +/-2 cm)
- [ ] Performance optimization (Dask tuning, target: <24h full pipeline)
- [ ] User guide and documentation finalization

---

## Documentation Status

| Document | Status |
|----------|--------|
| `docs/IMPLEMENTATION_STATUS.md` | Written |
| `docs/IMPLEMENTATION_SUMMARY.md` | Written |
| `docs/QUICKSTART.md` | Written |
| `docs/ML_TRAINING_GUIDE.md` | Written |
| `docs/GNSS_DATA_SOURCES.md` | Written |
| `docs/SETUP_MAC_M_CHIPS.md` | Written |

**Note:** Docs reference paths as `utils/internal/` but actual code lives under `libs/internal/`. Docs need manual.

---

## Summary

| Phase | # | Component | ai | test | manual | integration |
|-------|---|-----------|:--:|:----:|:----------:|:-----------:|
| 1 | 1a | GNSS ZTD Downloader | x | x | x | - |
| 1 | 1b | GNSS Position Downloader | - | - | - | - |
| 1 | 2 | InSAR/SBAS Processor | x | - | - | - |
| 1 | 3 | SEVIRI Downloader | x | - | - | - |
| 1 | 4 | ERA5 Vertical manual | - | - | - | - |
| 1 | 5 | GACOS manual | x | - | - | - |
| 2 | 6 | SEVIRI Temporal | x | - | - | - |
| 2 | 7 | SEVIRI SAR Geometry | x | - | - | - |
| 3 | 8 | ML Data Preparation | x | - | - | - |
| 3 | 9 | UNet Model + Trainer | x | - | - | - |
| 3 | 10 | Unit & Integration Tests | - | - | - | - |
| 4 | 11 | Pipeline Validation | - | - | - | - |

**Overall:** AI code generation done for most components. Component 1a (GNSS ZTD) is the most advanced (tested + corrected). Two new components pending: GNSS position data downloader (1b) and ERA5 vertical manual for SEVIRI (4). InSAR processor and SEVIRI downloader (pre-existing code) added to tracking.
