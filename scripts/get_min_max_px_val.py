from pathlib import Path

import numpy as np
import rasterio


def print_channel_min_max(path):
    """Print valid-pixel minimum and maximum across TIFFs in each channel folder."""
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {root}")

    image_paths = sorted(
        image_path
        for image_path in root.rglob("*")
        if image_path.is_file() and image_path.suffix.lower() in {".tif", ".tiff"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No TIFF images found in: {root}")

    channel_ranges = {}
    for image_path in image_paths:
        relative_path = image_path.relative_to(root)
        channel = relative_path.parts[0] if len(relative_path.parts) > 1 else root.name

        with rasterio.open(image_path) as image:
            pixels = np.ma.masked_invalid(image.read(masked=True))

        if pixels.count() == 0:
            raise ValueError(f"No valid pixels found in image: {image_path}")

        image_min = float(pixels.min())
        image_max = float(pixels.max())
        if channel in channel_ranges:
            current_min, current_max = channel_ranges[channel]
            channel_ranges[channel] = (
                min(current_min, image_min),
                max(current_max, image_max),
            )
        else:
            channel_ranges[channel] = (image_min, image_max)

    for channel, (minimum, maximum) in sorted(channel_ranges.items()):
        print(f"{channel}: min={minimum}, max={maximum}")

    return channel_ranges
