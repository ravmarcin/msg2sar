# ML Training Guide: Atmospheric Correction

Complete guide for training and using the ML-based atmospheric correction model.

**Optimized for:** Python 3.12 and Apple Silicon (Mac M1/M2/M3/M4)

---

## Quick Start

### 1. Test Your Setup

```bash
# Verify Apple Silicon (MPS) support
python scripts/test_model_mps.py
```

### 2. Test Model Architecture

```bash
python scripts/train_atmospheric_correction.py \
    --config data/configs/ml/atmospheric_correction/bogo_pl_master.json \
    --test-only
```

### 3. Prepare Training Data

```bash
python scripts/prepare_ml_data.py \
    --config data/configs/ml/atmospheric_correction/bogo_pl_master.json
```

### 4. Train Model

```bash
python scripts/train_atmospheric_correction.py \
    --config data/configs/ml/atmospheric_correction/bogo_pl_master.json
```

### 5. Run Inference

```bash
python scripts/inference_atmospheric_correction.py \
    --config data/configs/ml/atmospheric_correction/bogo_pl_master.json \
    --checkpoint checkpoints/best_model.pth \
    --input path/to/input.nc \
    --output path/to/output.nc
```

### 6. Monitor Training

```bash
tensorboard --logdir data/ml/atmospheric_correction/2023/bogo_pl/logs
```

Open browser to: http://localhost:6006

---

## Complete Workflow

### Step 1: Data Preparation

The model requires three types of input data:
1. **SEVIRI temporal stack** - Multi-channel satellite imagery
2. **SAR coherence maps** - Interferogram quality indicators
3. **GACOS-corrected phase** - Target for training

#### 1a. Process SEVIRI Data

```python
from utils.internal.msg.seviri_temporal import SeviriTemporalProcessor
from utils.internal.msg.msg_config import MsgConfig
from datetime import datetime

# Initialize
config = MsgConfig('data/configs/msg/seviri/2023/bogo_pl.json')
processor = SeviriTemporalProcessor(config)

# Process for each SAR timestamp
sar_timestamps = [
    datetime(2023, 6, 7, 4, 44, 45),
    datetime(2023, 7, 1, 4, 44, 46),
    # ... more timestamps
]

for timestamp in sar_timestamps:
    temporal_stack = processor.process_temporal_stack(
        stack_base=msg_stack,
        sar_timestamp=timestamp,
        seviri_data_dict=seviri_channels,
        n_images=10
    )
    processor.save_temporal_stack(temporal_stack, f'stack_{timestamp.strftime("%Y%m%d")}.nc')
```

#### 1b. Extract SAR Coherence

```python
# From SBAS processing
from utils.internal.sbas.sbas_process import SbasProcessor

processor = SbasProcessor('data/configs/sar/sbas/desc/2023/bogo_pl.json')
# Extract coherence maps from interferograms
# Save as coherence_maps.nc
```

#### 1c. Apply GACOS Corrections

```python
from utils.internal.gacos import GacosProcessor
from datetime import datetime

processor = GacosProcessor('data/configs/gacos/2023/bogo_pl.json')

for master_date, slave_date in interferogram_pairs:
    corrected = processor.process_interferogram(
        interferogram=intf_data,
        master_date=master_date,
        slave_date=slave_date,
        los_inc_angle=inc_angle_data,
        output_filename=f'corrected_{master_date}_{slave_date}.nc'
    )
```

#### 1d. Combine and Split Data

```bash
python scripts/prepare_ml_data.py \
    --config data/configs/ml/atmospheric_correction/bogo_pl_master.json
```

This script will:
- Load SEVIRI, coherence, and phase data
- Split into 80% training, 20% validation
- Compute normalization statistics
- Save split datasets

**Output:**
```
data/ml/atmospheric_correction/2023/bogo_pl/
├── training/
│   ├── seviri_temporal_stack.nc
│   ├── coherence_maps.nc
│   └── gacos_corrected_phase.nc
├── validation/
│   ├── seviri_temporal_stack.nc
│   ├── coherence_maps.nc
│   └── gacos_corrected_phase.nc
└── preprocessed/
    └── normalization_stats.json
```

---

### Step 2: Model Training

#### Configuration

Edit `data/configs/ml/atmospheric_correction/bogo_pl_master.json`:

```json
{
  "ml_training": {
    "batch_size": 16,              // Adjust for your RAM
    "train_val_split": 0.8,
    "num_workers": 4,               // Match CPU cores
    "seed": 42,
    "data_augmentation": {
      "random_flip": true,
      "random_rotation": 0,
      "noise_std": 0.01
    },
    "normalization": {
      "method": "standardize",      // or "minmax"
      "per_channel": true
    },
    "input_size": [256, 256],
    "output_size": [128, 128]
  },
  "ml_model": {
    "architecture": "unet",
    "in_channels": 14,
    "out_channels": 1,
    "init_features": 64,            // 32 for smaller model
    "input_size": 256,
    "output_size": 128,
    "loss_function": "mse",         // or "mae", "smooth_l1"
    "optimizer": {
      "type": "adam",               // or "adamw", "sgd"
      "lr": 0.0001,
      "weight_decay": 0.0001
    },
    "scheduler": {
      "type": "reduce_on_plateau",  // or "cosine", "step"
      "patience": 5,
      "factor": 0.5,
      "min_lr": 1e-7
    },
    "training": {
      "epochs": 100,
      "early_stopping_patience": 10,
      "checkpoint_frequency": 5,
      "save_best_only": true
    }
  }
}
```

#### Start Training

**Basic:**
```bash
python scripts/train_atmospheric_correction.py \
    --config data/configs/ml/atmospheric_correction/bogo_pl_master.json
```

**Resume from checkpoint:**
```bash
python scripts/train_atmospheric_correction.py \
    --config data/configs/ml/atmospheric_correction/bogo_pl_master.json \
    --resume checkpoint_epoch_20.pth
```

**Force CPU (for testing):**
```bash
python scripts/train_atmospheric_correction.py \
    --config data/configs/ml/atmospheric_correction/bogo_pl_master.json \
    --device cpu
```

#### Training Output

```
================================================================================
Atmospheric Correction Model Training
================================================================================
Job: bogo_pl_atmospheric_correction
Start time: 2026-04-23 14:30:00

Loading data...
✓ Data loaders created successfully

Initializing model...
✓ Model created with 31,042,881 parameters

Initializing trainer...
Using device: mps
✓ Trainer initialized

================================================================================
Starting training...
================================================================================

Epoch 1/100 | Train Loss: 0.523456 | Val Loss: 0.498321 | LR: 1.00e-04 | Time: 45.23s
✓ New best model saved (val_loss: 0.498321)

Epoch 2/100 | Train Loss: 0.456789 | Val Loss: 0.445123 | LR: 1.00e-04 | Time: 44.87s
✓ New best model saved (val_loss: 0.445123)

...

Epoch 50/100 | Train Loss: 0.123456 | Val Loss: 0.134567 | LR: 5.00e-06 | Time: 43.12s
Early stopping triggered after 50 epochs (10 epochs without improvement)

Training completed!
  Total time: 37.52 minutes
  Best val loss: 0.123456
  Final epoch: 50

================================================================================
Training Summary
================================================================================
Best validation loss: 0.123456
Final epoch: 50
Checkpoints saved to: data/ml/atmospheric_correction/2023/bogo_pl/checkpoints
TensorBoard logs: data/ml/atmospheric_correction/2023/bogo_pl/logs

To visualize training:
  tensorboard --logdir data/ml/atmospheric_correction/2023/bogo_pl/logs
```

---

### Step 3: Monitor Training

#### TensorBoard

```bash
tensorboard --logdir data/ml/atmospheric_correction/2023/bogo_pl/logs
```

**Metrics to watch:**
- **Loss/train**: Should decrease steadily
- **Loss/val**: Should decrease, may plateau
- **Learning_Rate**: Should decrease if using scheduler
- **Gradients**: Check for vanishing/exploding gradients

#### Activity Monitor (Mac)

1. Open Activity Monitor
2. **Memory tab**: Ensure no swapping (Memory Pressure should be green)
3. **GPU History**: Verify GPU is being used
4. **CPU tab**: Check CPU usage is reasonable

#### Expected Training Time

| Device | Dataset Size | Time per Epoch | Total Time (50 epochs) |
|--------|--------------|----------------|------------------------|
| M1 Pro | 1000 samples | ~45s | ~37 min |
| M1 Max | 1000 samples | ~35s | ~29 min |
| M2 Pro | 1000 samples | ~40s | ~33 min |
| M2 Max | 1000 samples | ~30s | ~25 min |
| M3 Pro | 1000 samples | ~35s | ~29 min |
| M3 Max | 1000 samples | ~25s | ~21 min |

*Times are approximate and vary with batch size and model complexity.*

---

### Step 4: Inference

#### Single File

```bash
python scripts/inference_atmospheric_correction.py \
    --config data/configs/ml/atmospheric_correction/bogo_pl_master.json \
    --checkpoint checkpoints/best_model.pth \
    --input data/ml/atmospheric_correction/2023/bogo_pl/validation/seviri_temporal_stack.nc \
    --output predictions/atmospheric_corrections.nc
```

#### Batch Processing

```python
import glob
from pathlib import Path

# Process all validation files
input_files = glob.glob('data/validation/*.nc')

for input_file in input_files:
    output_file = Path(input_file).stem + '_corrected.nc'

    # Run inference
    # ... (use inference script or API)
```

#### Python API

```python
import torch
from utils.internal.ml import AtmosphericCorrectionUNet, MLDataConfig

# Load config
config = MLDataConfig('path/to/config.json')

# Load model
checkpoint = torch.load('checkpoints/best_model.pth', map_location='mps')
model = AtmosphericCorrectionUNet(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Move to device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)

# Inference
with torch.no_grad():
    input_tensor = torch.randn(1, 14, 256, 256).to(device)
    output = model(input_tensor)

print(f"Output shape: {output.shape}")  # [1, 1, 128, 128]
```

---

## Troubleshooting

### Training Issues

#### 1. Out of Memory

**Error:**
```
RuntimeError: MPS backend out of memory
```

**Solutions:**
1. Reduce batch size: `"batch_size": 8` or `4`
2. Reduce model size: `"init_features": 32`
3. Clear cache:
   ```python
   torch.mps.empty_cache()
   ```
4. Close other applications
5. Restart your Mac

#### 2. Loss Not Decreasing

**Symptoms:**
- Training loss remains high
- Validation loss not improving

**Solutions:**
1. Check learning rate (try `1e-3` or `1e-5`)
2. Check data normalization (verify statistics)
3. Check data loading (ensure no NaN values)
4. Try different optimizer (AdamW instead of Adam)
5. Increase model capacity: `"init_features": 128`

#### 3. Overfitting

**Symptoms:**
- Training loss very low
- Validation loss high/increasing

**Solutions:**
1. Increase data augmentation
2. Add dropout (modify model)
3. Reduce model capacity
4. Early stopping (already enabled)
5. Collect more training data

#### 4. Slow Training

**Symptoms:**
- Taking much longer than expected
- Low GPU usage

**Solutions:**
1. Verify MPS is being used:
   ```bash
   # Check logs for "Using device: mps"
   ```
2. Reduce `num_workers` if CPU-bound
3. Increase batch size (if memory allows)
4. Check for swapping in Activity Monitor
5. Ensure data is preprocessed (not loading raw data)

### MPS-Specific Issues

#### Operation Not Supported

**Error:**
```
NotImplementedError: The operator 'aten::...' is not currently implemented for the MPS device
```

**Solutions:**
1. Update PyTorch: `pip install --upgrade torch`
2. Use CPU for unsupported operation:
   ```python
   # Temporarily move to CPU
   x_cpu = x.to('cpu')
   result = unsupported_operation(x_cpu)
   result = result.to('mps')
   ```
3. File issue with PyTorch team

#### MPS Synchronization

If results seem incorrect, ensure synchronization:
```python
if device.type == 'mps':
    torch.mps.synchronize()
```

---

## Performance Optimization

### Optimal Settings by Device

#### M1/M2 Pro (16 GB)
```json
{
  "ml_training": {
    "batch_size": 12,
    "num_workers": 4
  },
  "ml_model": {
    "init_features": 64
  }
}
```

#### M1/M2 Max (32 GB)
```json
{
  "ml_training": {
    "batch_size": 24,
    "num_workers": 6
  },
  "ml_model": {
    "init_features": 96
  }
}
```

#### M3 Pro/Max (36+ GB)
```json
{
  "ml_training": {
    "batch_size": 32,
    "num_workers": 8
  },
  "ml_model": {
    "init_features": 128
  }
}
```

### Memory Management

```python
# Monitor memory
import psutil
mem = psutil.virtual_memory()
print(f"Available: {mem.available / (1024**3):.1f} GB")

# Clear cache periodically
if epoch % 10 == 0:
    torch.mps.empty_cache()
```

### Profiling

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
    for inputs, targets in train_loader:
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward()
        break

print(prof.key_averages().table(sort_by="self_cpu_time_total"))
```

---

## Model Architecture Details

### UNet Structure

```
Input: [14, 256, 256]
│
├─ Encoder Level 1: Conv → BN → ReLU (64 channels)
│  ├─ Skip Connection 1 ──────────────────┐
│  └─ MaxPool (128×128)                   │
│                                          │
├─ Encoder Level 2: Conv → BN → ReLU (128 channels)
│  ├─ Skip Connection 2 ────────────┐     │
│  └─ MaxPool (64×64)               │     │
│                                    │     │
├─ Encoder Level 3: Conv → BN → ReLU (256 channels)
│  ├─ Skip Connection 3 ──────┐     │     │
│  └─ MaxPool (32×32)         │     │     │
│                              │     │     │
├─ Encoder Level 4: Conv → BN → ReLU (512 channels)
│  ├─ Skip Connection 4 ──┐   │     │     │
│  └─ MaxPool (16×16)     │   │     │     │
│                          │   │     │     │
├─ Bottleneck: Conv → BN → ReLU (1024 channels, 8×8)
│                          │   │     │     │
├─ Decoder Level 1: Up → Concat ◄─┘   │     │     │
│  └─ Conv → BN → ReLU (512 channels, 16×16)
│                              │     │     │
├─ Decoder Level 2: Up → Concat ◄────┘     │     │
│  └─ Conv → BN → ReLU (256 channels, 32×32)
│                                    │     │
├─ Decoder Level 3: Up → Concat ◄─────────┘     │
│  └─ Conv → BN → ReLU (128 channels, 64×64)
│                                          │
├─ Decoder Level 4: Up → Concat ◄──────────────┘
│  └─ Conv → BN → ReLU (64 channels, 128×128)
│
└─ Output: 1×1 Conv → Center Crop
   Output: [1, 128, 128]
```

### Parameter Count

| Component | Parameters |
|-----------|------------|
| Encoder | ~13M |
| Bottleneck | ~5M |
| Decoder | ~13M |
| **Total** | **~31M** |

---

## Validation and Results

### Metrics

After training, evaluate model performance:

```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load predictions and ground truth
predictions = load_predictions('predictions.nc')
ground_truth = load_ground_truth('gacos_corrected_phase.nc')

# Compute metrics
mse = mean_squared_error(ground_truth.flatten(), predictions.flatten())
rmse = np.sqrt(mse)
r2 = r2_score(ground_truth.flatten(), predictions.flatten())

print(f"RMSE: {rmse:.4f} rad")
print(f"R²: {r2:.4f}")
```

### Success Criteria

- **RMSE < 0.5 rad** (good correction)
- **R² > 0.7** (strong correlation with GACOS)
- **GNSS validation within ±2 cm** (ZTD)

### Visualization

```python
import matplotlib.pyplot as plt

# Plot predictions vs ground truth
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(ground_truth[0], cmap='RdBu')
plt.title('Ground Truth (GACOS)')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(predictions[0], cmap='RdBu')
plt.title('ML Prediction')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(ground_truth[0] - predictions[0], cmap='RdBu')
plt.title('Residual')
plt.colorbar()

plt.tight_layout()
plt.savefig('correction_comparison.png')
```

---

## Advanced Topics

### Custom Loss Functions

```python
# In trainer.py, add custom loss
class WeightedMSELoss(nn.Module):
    def __init__(self, coherence_weight=0.5):
        super().__init__()
        self.coherence_weight = coherence_weight

    def forward(self, pred, target, coherence):
        weights = coherence ** self.coherence_weight
        return torch.mean(weights * (pred - target)**2)
```

### Model Ensemble

```python
# Load multiple models
models = [
    load_model('checkpoint_epoch_30.pth'),
    load_model('checkpoint_epoch_40.pth'),
    load_model('checkpoint_epoch_50.pth')
]

# Ensemble prediction
with torch.no_grad():
    predictions = [model(input_tensor) for model in models]
    ensemble_prediction = torch.stack(predictions).mean(dim=0)
```

### Transfer Learning

```python
# Load pretrained model
checkpoint = torch.load('pretrained_model.pth')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# Freeze encoder
for name, param in model.named_parameters():
    if 'enc' in name or 'inc' in name:
        param.requires_grad = False

# Fine-tune decoder only
```

---

## Additional Resources

- **Model Code:** `utils/internal/ml/models/unet.py`
- **Trainer Code:** `utils/internal/ml/trainer.py`
- **Data Loader:** `utils/internal/ml/data_loader.py`
- **Setup Guide:** `SETUP_MAC_M_CHIPS.md`
- **Implementation Status:** `IMPLEMENTATION_STATUS.md`

---

**Last Updated:** 2026-04-23
**Python Version:** 3.12+
**PyTorch Version:** 2.2.0+
**Device Support:** Apple Silicon (MPS), CUDA, CPU
