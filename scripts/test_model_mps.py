#!/usr/bin/env python3
"""
Quick test script for Apple Silicon (MPS) compatibility.

Tests:
1. MPS availability
2. Model initialization
3. Forward pass
4. Backward pass
5. Memory usage
6. Performance benchmark

Usage:
    python scripts/test_model_mps.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError:
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

import torch
import time
import numpy as np
from utils.internal.ml.models.unet import AtmosphericCorrectionUNet


def test_mps_availability():
    """Test if MPS is available."""
    print("="*80)
    print("1. Testing MPS Availability")
    print("="*80)

    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        print("✓ MPS is available and ready to use!")
        return True
    else:
        print("✗ MPS is not available. Using CPU.")
        return False


def test_model_creation(device):
    """Test model creation and move to device."""
    print("\n" + "="*80)
    print("2. Testing Model Creation")
    print("="*80)

    try:
        model = AtmosphericCorrectionUNet(
            in_channels=14,
            out_channels=1,
            init_features=64,
            input_size=256,
            output_size=128
        )

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created successfully")
        print(f"Total parameters: {total_params:,}")

        # Move to device
        model = model.to(device)
        print(f"Model moved to {device}")
        print("✓ Model creation test passed!")

        return model

    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return None


def test_forward_pass(model, device):
    """Test forward pass."""
    print("\n" + "="*80)
    print("3. Testing Forward Pass")
    print("="*80)

    try:
        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, 14, 256, 256).to(device)

        print(f"Input shape: {x.shape}")
        print(f"Input device: {x.device}")

        # Forward pass
        with torch.no_grad():
            start_time = time.time()
            y = model(x)
            forward_time = time.time() - start_time

        print(f"Output shape: {y.shape}")
        print(f"Output device: {y.device}")
        print(f"Forward pass time: {forward_time*1000:.2f} ms")

        # Verify output shape
        expected_shape = (batch_size, 1, 128, 128)
        if y.shape == expected_shape:
            print(f"✓ Output shape correct: {y.shape}")
        else:
            print(f"✗ Output shape incorrect: expected {expected_shape}, got {y.shape}")
            return False

        print("✓ Forward pass test passed!")
        return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass(model, device):
    """Test backward pass and gradient computation."""
    print("\n" + "="*80)
    print("4. Testing Backward Pass")
    print("="*80)

    try:
        # Create dummy data
        x = torch.randn(2, 14, 256, 256).to(device)
        target = torch.randn(2, 1, 128, 128).to(device)

        # Forward pass
        output = model(x)

        # Compute loss
        criterion = torch.nn.MSELoss()
        loss = criterion(output, target)

        print(f"Loss value: {loss.item():.6f}")

        # Backward pass
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time

        print(f"Backward pass time: {backward_time*1000:.2f} ms")

        # Check gradients
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                break

        if has_gradients:
            print("✓ Gradients computed successfully")
        else:
            print("✗ No gradients found")
            return False

        print("✓ Backward pass test passed!")
        return True

    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage(device):
    """Test memory usage."""
    print("\n" + "="*80)
    print("5. Testing Memory Usage")
    print("="*80)

    try:
        import psutil

        # Get system memory
        mem = psutil.virtual_memory()
        print(f"System memory:")
        print(f"  Total: {mem.total / (1024**3):.1f} GB")
        print(f"  Available: {mem.available / (1024**3):.1f} GB")
        print(f"  Used: {mem.percent:.1f}%")

        # Test with different batch sizes
        batch_sizes = [1, 2, 4, 8, 16]

        print("\nTesting different batch sizes:")
        for bs in batch_sizes:
            try:
                model = AtmosphericCorrectionUNet(
                    in_channels=14,
                    out_channels=1,
                    init_features=64,
                    input_size=256,
                    output_size=128
                ).to(device)

                x = torch.randn(bs, 14, 256, 256).to(device)

                with torch.no_grad():
                    y = model(x)

                mem_after = psutil.virtual_memory()
                print(f"  Batch size {bs:2d}: ✓ OK (Memory: {mem_after.percent:.1f}%)")

                # Clean up
                del model, x, y
                if device.type == 'mps':
                    torch.mps.empty_cache()
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  Batch size {bs:2d}: ✗ Out of memory")
                    break
                else:
                    raise

        print("✓ Memory test completed!")
        return True

    except Exception as e:
        print(f"✗ Memory test failed: {e}")
        return False


def benchmark_performance(device):
    """Benchmark performance."""
    print("\n" + "="*80)
    print("6. Performance Benchmark")
    print("="*80)

    try:
        model = AtmosphericCorrectionUNet(
            in_channels=14,
            out_channels=1,
            init_features=64,
            input_size=256,
            output_size=128
        ).to(device)

        model.eval()

        batch_size = 8
        num_iterations = 20

        # Warmup
        print("Warming up...")
        x = torch.randn(batch_size, 14, 256, 256).to(device)
        for _ in range(5):
            with torch.no_grad():
                _ = model(x)

        # Benchmark forward pass
        print(f"\nBenchmarking forward pass ({num_iterations} iterations, batch_size={batch_size})...")
        times = []

        for i in range(num_iterations):
            x = torch.randn(batch_size, 14, 256, 256).to(device)

            start_time = time.time()
            with torch.no_grad():
                y = model(x)

            # Synchronize if using MPS
            if device.type == 'mps':
                torch.mps.synchronize()

            elapsed = time.time() - start_time
            times.append(elapsed)

        times = np.array(times)
        mean_time = times.mean()
        std_time = times.std()

        print(f"\nResults:")
        print(f"  Mean time per batch: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"  Throughput: {batch_size/mean_time:.1f} samples/sec")
        print(f"  Min time: {times.min()*1000:.2f} ms")
        print(f"  Max time: {times.max()*1000:.2f} ms")

        print("✓ Benchmark completed!")
        return True

    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MSG2SAR Apple Silicon (MPS) Test Suite")
    print("="*80 + "\n")

    # Test 1: MPS availability
    mps_available = test_mps_availability()

    # Select device
    if mps_available:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\nUsing device: {device}")

    # Test 2: Model creation
    model = test_model_creation(device)
    if model is None:
        print("\n✗ Tests failed at model creation")
        return

    # Test 3: Forward pass
    if not test_forward_pass(model, device):
        print("\n✗ Tests failed at forward pass")
        return

    # Test 4: Backward pass
    if not test_backward_pass(model, device):
        print("\n✗ Tests failed at backward pass")
        return

    # Test 5: Memory usage
    test_memory_usage(device)

    # Test 6: Performance benchmark
    benchmark_performance(device)

    # Final summary
    print("\n" + "="*80)
    print("All Tests Completed!")
    print("="*80)
    print("\n✓ Your system is ready for training!")
    print("\nNext steps:")
    print("  1. Prepare your data: python scripts/prepare_ml_data.py --config <config>")
    print("  2. Start training: python scripts/train_atmospheric_correction.py --config <config>")
    print("  3. Monitor with TensorBoard: tensorboard --logdir <log_dir>")


if __name__ == "__main__":
    main()
