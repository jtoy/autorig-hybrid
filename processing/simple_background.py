from PIL import Image, ImageDraw, ImageFilter
import numpy as np


def remove_background_simple(image_path: str, tolerance: int = 30):
    """
    Removes the background using flood-fill from corners and cleans up edges.
    """
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size

    rgb_image = image.convert("RGB")

    # Flood-fill from ALL edge pixels to find background region.
    # Seeding from only the 4 corners fails when the subject fills a corner.
    # By sampling every pixel on all 4 edges we guarantee that any white
    # background pixel reachable from the border gets marked.
    work = rgb_image.copy()
    marker = (254, 0, 254)
    arr_init = np.array(work)
    # Collect edge pixels that are near-white (within tolerance of 255,255,255)
    edge_seeds = set()
    for x in range(width):
        for y in [0, height - 1]:
            r, g, b = arr_init[y, x]
            if abs(int(r) - 255) <= tolerance and abs(int(g) - 255) <= tolerance and abs(int(b) - 255) <= tolerance:
                edge_seeds.add((x, y))
    for y in range(height):
        for x in [0, width - 1]:
            r, g, b = arr_init[y, x]
            if abs(int(r) - 255) <= tolerance and abs(int(g) - 255) <= tolerance and abs(int(b) - 255) <= tolerance:
                edge_seeds.add((x, y))
    for seed in edge_seeds:
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
