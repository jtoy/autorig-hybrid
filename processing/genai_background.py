import os
from .background import change_background
from .alpha import change_alpha

def remove_background_genai(client, image_path: str, tolerance: int = 30):
    """
    Removes background using GenAI by generating black/white backgrounds and merging alpha.
    Overwrites image_path in place.

    Args:
        client: Google GenAI client instance.
        image_path: Path to the PNG image to process (overwritten in place).
        tolerance: Threshold used by alpha merge (0-255).
    """
    base_name, _ = os.path.splitext(image_path)
    black_path = f"{base_name}_black.png"
    white_path = f"{base_name}_white.png"

    change_background(client, image_path, "black", black_path)
    change_background(client, image_path, "white", white_path)
    change_alpha(black_path, white_path, image_path)

    if os.path.exists(black_path):
        os.remove(black_path)
    if os.path.exists(white_path):
        os.remove(white_path)