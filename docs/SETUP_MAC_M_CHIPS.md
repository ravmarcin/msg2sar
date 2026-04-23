# Setup Guide for Mac M Chips (Apple Silicon) with Python 3.12

This guide covers setup and optimization for running MSG2SAR atmospheric correction on Mac M1/M2/M3 chips with Python 3.12.

---

## System Requirements

- **Hardware:** Mac with M1, M2, M3, or M4 chip (Apple Silicon)
- **OS:** macOS 12.0 (Monterey) or later
- **Python:** 3.12.x
- **RAM:** 16 GB minimum, 32 GB recommended for large datasets

---

## Installation Steps

### 1. Install Conda (Miniforge for ARM64)

Miniforge provides native ARM64 support for Apple Silicon:

```bash
# Download Miniforge3 for macOS ARM64
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh

# Install
bash Miniforge3-MacOSX-arm64.sh

# Follow the prompts, restart terminal when done
```

### 2. Create Conda Environment

```bash
# Create environment with Python 3.12
conda create -n msg2sar python=3.12 -y

# Activate environment
conda activate msg2sar
```

### 3. Install System Dependencies

```bash
# Install required conda packages
conda install -c conda-forge numpy scipy pandas xarray netcdf4 rasterio geopandas shapely -y

# Install GDAL and geospatial libraries
conda install -c conda-forge gdal pyproj -y
```

### 4. Install PyTorch with MPS Support

PyTorch 2.2+ has native support for Apple's Metal Performance Shaders (MPS):

```bash
# Install PyTorch for Apple Silicon
pip install torch>=2.2.0 torchvision>=0.17.0 --index-url https://download.pytorch.org/whl/cpu
```

**Note:** PyTorch will automatically detect and use MPS backend on Apple Silicon.

### 5. Install Project Dependencies

```bash
# Navigate to project root
cd /path/to/msg2sar

# Install pip requirements
pip install -r env_build/pip_requirements.txt
```

### 6. Verify MPS Support

Test that PyTorch can use the MPS backend:

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available()); print('MPS built:', torch.backends.mps.is_built())"
```

Expected output:
```
MPS available: True
MPS built: True
```

---

## Performance Optimization for Apple Silicon

### GPU Acceleration (MPS)

The Metal Performance Shaders (MPS) backend provides GPU acceleration on Apple Silicon:

**Advantages:**
- Native GPU acceleration without CUDA
- Optimized for unified memory architecture
- Lower power consumption than discrete GPUs
- Automatic memory management

**Limitations:**
- Some PyTorch operations not yet supported (falls back to CPU)
- Different performance characteristics than CUDA
- Memory shared with system (unified memory)

### Memory Management

Apple Silicon uses unified memory (shared between CPU and GPU):

```python
# Check available memory
import psutil
available_gb = psutil.virtual_memory().available / (1024**3)
print(f"Available memory: {available_gb:.1f} GB")

# Adjust batch size based on available memory
# For 16 GB: batch_size = 8-16
# For 32 GB: batch_size = 16-32
```

### Optimal Settings

In your config file (`bogo_pl_master.json`):

```json
{
  "ml_training": {
    "batch_size": 16,        // Adjust based on RAM (8-32)
    "num_workers": 4,        // Match performance cores (M1: 4, M2: 4-6, M3: 6-8)
    "pin_memory": false      // Not needed on unified memory
  }
}
```

### CPU vs MPS Performance

**When to use MPS:**
- Large batch sizes (>8)
- Complex models (UNet with many features)
- Inference on multiple samples

**When MPS may be slower:**
- Very small batches (<4)
- Simple operations
- Frequent CPU-GPU transfers

**Force CPU if needed:**
```bash
python scripts/train_atmospheric_correction.py --config config.json --device cpu
```

---

## Testing Your Setup

### 1. Test Model Architecture

```bash
python scripts/train_atmospheric_correction.py \
    --config data/configs/ml/atmospheric_correction/bogo_pl_master.json \
    --test-only
```

Expected output:
```
Using device: mps
✓ Forward pass successful!
✓ Backward pass successful!
Model test completed successfully!
```

### 2. Test Model Training (Quick)

Create a minimal test to verify training works:

```python
# test_training.py
import torch
from utils.internal.ml.models.unet import AtmosphericCorrectionUNet

# Check device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Create small model
model = AtmosphericCorrectionUNet(
    in_channels=14,
    out_channels=1,
    init_features=32,  # Reduced for testing
    input_size=256,
    output_size=128
).to(device)

# Test forward/backward
x = torch.randn(2, 14, 256, 256).to(device)
y = model(x)
loss = torch.nn.MSELoss()(y, torch.randn_like(y))
loss.backward()

print(f"✓ Test passed! Output shape: {y.shape}")
```

Run:
```bash
python test_training.py
```

### 3. Benchmark Performance

```bash
# Run benchmark script
python scripts/benchmark_device.py
```

---

## Common Issues and Solutions

### Issue 1: MPS Not Available

**Symptom:**
```
MPS available: False
```

**Solutions:**
1. Ensure macOS >= 12.3
2. Update PyTorch: `pip install --upgrade torch`
3. Check PyTorch was installed correctly (not CPU-only version)

### Issue 2: Out of Memory

**Symptom:**
```
RuntimeError: MPS backend out of memory
```

**Solutions:**
1. Reduce batch size in config
2. Reduce model features (`init_features=32` instead of 64)
3. Close other applications
4. Restart to clear memory

### Issue 3: Slow Performance

**Symptom:** Training slower than expected

**Solutions:**
1. Verify MPS is being used: check logs for "Using device: mps"
2. Reduce `num_workers` if CPU-bound (try 2-4)
3. Ensure no swapping: check Activity Monitor
4. Try smaller input sizes first

### Issue 4: Unsupported Operations

**Symptom:**
```
NotImplementedError: The operator 'aten::...' is not currently implemented for the MPS device
```

**Solutions:**
1. Update PyTorch to latest version
2. Fallback to CPU for that operation:
   ```python
   if operation_not_supported:
       tensor = tensor.to('cpu')
       result = operation(tensor)
       result = result.to('mps')
   ```
3. File an issue with PyTorch if operation is critical

---

## Performance Comparison

### Training Speed (relative to NVIDIA RTX 3090)

| Device | UNet Training | Inference | Power |
|--------|--------------|-----------|-------|
| **M1 Pro (16 GB)** | 0.5-0.7x | 0.6-0.8x | 15-20W |
| **M1 Max (32 GB)** | 0.6-0.8x | 0.7-0.9x | 20-30W |
| **M2 Pro** | 0.6-0.8x | 0.7-0.9x | 15-25W |
| **M2 Max** | 0.7-0.9x | 0.8-1.0x | 20-35W |
| **M3 Pro** | 0.7-0.9x | 0.8-1.0x | 15-25W |
| **M3 Max** | 0.8-1.0x | 0.9-1.1x | 20-40W |

*Note: Performance varies by workload. M3 Max can match or exceed RTX 3090 for some operations.*

### Memory Efficiency

| RAM | Recommended Batch Size | Max Model Size |
|-----|----------------------|----------------|
| 16 GB | 8-16 | ~200M params |
| 32 GB | 16-32 | ~500M params |
| 64 GB | 32-64 | ~1B params |

---

## Monitoring Tools

### Activity Monitor

Monitor resource usage:
1. Open Activity Monitor
2. Check **Memory** tab: ensure not swapping
3. Check **GPU History**: see GPU usage
4. Check **Energy**: verify power efficiency

### TensorBoard

Monitor training progress:
```bash
# Start TensorBoard
tensorboard --logdir data/ml/atmospheric_correction/2023/bogo_pl/logs

# Open browser to http://localhost:6006
```

### PyTorch Profiler

Profile model performance:
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
    output = model(input)

print(prof.key_averages().table(sort_by="self_cpu_time_total"))
```

---

## Recommended Workflow

### 1. Start Small
```json
{
  "ml_training": {
    "batch_size": 8,
    "epochs": 10
  },
  "ml_model": {
    "init_features": 32
  }
}
```

### 2. Scale Up Gradually
- Increase batch size until memory is ~80% used
- Increase model size if performance allows
- Monitor for swapping (should be minimal)

### 3. Production Settings
```json
{
  "ml_training": {
    "batch_size": 16,      // Optimal for most M chips
    "num_workers": 4,      // Match performance cores
    "epochs": 100
  },
  "ml_model": {
    "init_features": 64    // Full model
  }
}
```

---

## Additional Resources

- **PyTorch MPS Docs:** https://pytorch.org/docs/stable/notes/mps.html
- **Apple ML Docs:** https://developer.apple.com/metal/
- **Optimization Guide:** https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

---

## Troubleshooting Commands

```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__); print('MPS:', torch.backends.mps.is_available())"

# Check available devices
python -c "import torch; print([torch.device(d) for d in ['cpu', 'mps'] if d == 'cpu' or torch.backends.mps.is_available()])"

# Test simple operation on MPS
python -c "import torch; x = torch.randn(10, 10).to('mps'); y = x @ x.T; print('MPS test passed')"

# Check memory usage
python -c "import psutil; print(f'Available: {psutil.virtual_memory().available/(1024**3):.1f} GB')"
```

---

**Last Updated:** 2026-04-23
**Supported Models:** M1, M2, M3, M4 series
**Python Version:** 3.12.x
**PyTorch Version:** 2.2.0+
