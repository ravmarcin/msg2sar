"""FILM based temporal interpolation for single-band satellite images."""

from pathlib import Path

import numpy as np


def prepare_film_input(image, vmin, vmax):
    """Normalize a physical single-band image to the RGB input expected by FILM."""
    if vmax <= vmin:
        raise ValueError("vmax must be greater than vmin")
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image, nan=vmin, posinf=vmax, neginf=vmin)
    image = np.clip(image, vmin, vmax)
    image = (image - vmin) / (vmax - vmin)
    return np.repeat(image[..., np.newaxis], 3, axis=-1)


def load_film_model(model_path):
    """Load the TensorFlow Hub SavedModel on demand."""
    try:
        import tensorflow_hub as hub
    except ImportError as exc:
        raise ImportError("FILM interpolation requires tensorflow-hub") from exc
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"FILM model not found: {path}")
    return hub.load(str(path))


def film_temporal_upsampling(image_before, image_after, fraction, vmin, vmax, model):
    """Interpolate one physical single-band image at a fraction of its interval."""
    fraction = float(fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    before = np.asarray(image_before)
    after = np.asarray(image_after)
    if before.ndim != 2 or after.ndim != 2 or before.shape != after.shape:
        raise ValueError("FILM inputs must be same-sized, two-dimensional arrays")

    # FILM is an interpolation model. Return the endpoints directly to avoid
    # model changes to an observed input at alpha 0 or 1.
    if fraction == 0.0:
        return before.astype(np.float32, copy=True)
    if fraction == 1.0:
        return after.astype(np.float32, copy=True)

    input_before = prepare_film_input(before, vmin, vmax)
    input_after = prepare_film_input(after, vmin, vmax)
    model_input = {
        "time": np.asarray([[fraction]], dtype=np.float32),
        "x0": input_before[np.newaxis, ...],
        "x1": input_after[np.newaxis, ...],
    }
    prediction = model(model_input)["image"][0].numpy()
    if prediction.ndim == 3:
        prediction = np.mean(prediction, axis=-1)
    prediction = np.clip(prediction, 0.0, 1.0)
    return (prediction * (vmax - vmin) + vmin).astype(np.float32)
