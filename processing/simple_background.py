from PIL import Image, ImageDraw, ImageFilter
import numpy as np


def remove_background_simple(image_path: str, tolerance: int = 30):
    """
    Removes the background using flood-fill from corners and cleans up edges.
    """
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size

    rgb_image = image.convert("RGB")

    # Flood-fill from all 4 corners to find background region
    # Use a working copy and fill with a marker color
    work = rgb_image.copy()
    marker = (254, 0, 254)
    for seed in [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]:
        ImageDraw.floodfill(work, seed, marker, thresh=tolerance)

    # Build mask from marker pixels using numpy for speed
    arr = np.array(work)
    bg_mask = (arr[:, :, 0] == 254) & (arr[:, :, 1] == 0) & (arr[:, :, 2] == 254)

    # Create alpha mask: 255 where background, 0 where foreground
    mask = Image.fromarray((bg_mask * 255).astype(np.uint8))

    # Clean up: dilate slightly to catch anti-aliased edges
    mask = mask.filter(ImageFilter.MaxFilter(3))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Apply mask to alpha channel
    img_arr = np.array(image)
    mask_arr = np.array(mask).astype(np.float32) / 255.0
    img_arr[:, :, 3] = (img_arr[:, :, 3].astype(np.float32) * (1.0 - mask_arr)).astype(np.uint8)

    result = Image.fromarray(img_arr)
    result.save(image_path)
