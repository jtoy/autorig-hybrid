import numpy as np
from PIL import Image


def change_alpha(black_path, white_path, output_path):
    """
    Combines an image with a black background and one with a white background
    to create a PNG with a transparent background using a two-pass algorithm.

    Args:
        black_path (str): Path to the image with a black background.
        white_path (str): Path to the image with a white background.
        output_path (str): Path where the resulting transparent PNG will be saved.
    """
    try:
        img_b = Image.open(black_path).convert("RGB")
        img_w = Image.open(white_path).convert("RGB")
    except FileNotFoundError as e:
        print(f"Error loading images: {e}")
        return None

    if img_b.size != img_w.size:
        print("Error: Images must have the same dimensions.")
        return None

    # Convert to Numpy arrays for calculation
    arr_b = np.array(img_b, dtype=np.float32)
    arr_w = np.array(img_w, dtype=np.float32)

    # Distance between pure White (255,255,255) and pure Black (0,0,0)
    bg_dist = np.sqrt(3 * 255**2)

    # Calculate the distance between the two observed pixels
    # pixel_dist = sqrt(sum((W - B)^2))
    diff = arr_w - arr_b
    pixel_dist = np.sqrt(np.sum(diff**2, axis=2))

    # Calculate alpha: 1 - (pixelDist / bgDist)
    alpha = 1.0 - (pixel_dist / bg_dist)
    alpha = np.clip(alpha, 0.0, 1.0)

    # Recover foreground color from the version on black
    # C_out = B / alpha
    alpha_expanded = np.expand_dims(alpha, axis=2)

    # Initialize output color with zeros
    color_out = np.zeros_like(arr_b)

    # Mask for non-transparent pixels
    mask = alpha > 0.01

    # Apply color recovery where alpha is high enough
    color_out[mask] = arr_b[mask] / alpha_expanded[mask]
    color_out = np.clip(color_out, 0, 255)

    # Merge channels
    r_out, g_out, b_out = [color_out[:, :, i].astype(np.uint8) for i in range(3)]
    a_out = (alpha * 255).astype(np.uint8)

    final_img = Image.merge(
        "RGBA",
        (
            Image.fromarray(r_out),
            Image.fromarray(g_out),
            Image.fromarray(b_out),
            Image.fromarray(a_out),
        ),
    )

    final_img.save(output_path, "PNG")
    return final_img