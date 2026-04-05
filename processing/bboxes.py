import os
import json
import io
import base64
from google.genai import types
from PIL import Image


def parse_json(json_output: str) -> str:
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i + 1:])
            json_output = json_output.split("```")[0]
            break
    return json_output

def bboxes(client, inputPath, outputFolder, model="gemini-3-flash-preview"):
    """
    Detects 14 parts of a figure and saves crops to outputFolder.

    Left/right for paired parts (arms, legs) is assigned purely by x-position
    in the image: the box with smaller center-x becomes left_*, the other right_*.

    Args:
        client: The Google GenAI client instance.
        inputPath: Path to the input image file.
        outputFolder: Path where the cropped images will be saved.
        model: Model to use for bounding box detection.
    """
    temperature = 0
    prompt = """
    Task: ONLY detect and return bounding boxes. Do not edit, redraw, or reinterpret the image.
    This image contains 14 SEPARATED body-part pieces laid out on a white background.
    Detect exactly these 14 parts using NEUTRAL labels (no left/right):
    head, torso, upperarm_1, upperarm_2, forearm_1, forearm_2, hand_1, hand_2, thigh_1, thigh_2, calf_1, calf_2, foot_1, foot_2.

    Use "_1" and "_2" suffixes to distinguish pairs without assuming left/right.

    CRITICAL arm/leg splitting rules:
    - Each arm is THREE separate pieces: upperarm (shoulder to elbow), forearm (elbow to wrist), and hand (wrist to fingertips).
    - Each leg is THREE separate pieces: thigh (hip to knee), calf (knee to ankle), and foot (ankle to toes).
    - Boxes should NOT significantly overlap.

    Return exactly 14 SEPARATE bounding boxes in [ymin, xmin, ymax, xmax] format normalized to 0-1000.
    Output a JSON array of objects, each with 'box_2d' and 'label'.
    """

    if not os.path.exists(inputPath):
        raise FileNotFoundError(f"Image file not found: {inputPath}")

    image = Image.open(inputPath)
    width, height = image.size

    print(f"[bboxes] Detecting 14 body parts with neutral labels using {model}...")
    response = client.models.generate_content(
        model=model,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=temperature,
        )
    )

    bounding_boxes = json.loads(response.text)
    items = bounding_boxes if isinstance(bounding_boxes, list) else bounding_boxes.get("items", [])
    print(f"[bboxes] Got {len(items)} bounding boxes")

    def _box_center_x(box):
        abs_x1 = int(box[1] / 1000 * width)
        abs_x2 = int(box[3] / 1000 * width)
        return (abs_x1 + abs_x2) / 2

    PAIRED_LABELS = {
        "upperarm_1", "upperarm_2", "forearm_1", "forearm_2", "hand_1", "hand_2",
        "thigh_1", "thigh_2", "calf_1", "calf_2", "foot_1", "foot_2",
    }
    PAIRED_BASES = ("upperarm", "forearm", "hand", "thigh", "calf", "foot")

    # Resolve labels: for paired parts, assign left/right purely by x-position
    # (smaller x = left_*, larger x = right_*).
    labeled = []
    for i, item in enumerate(items):
        label = item.get("label", f"part_{i}").replace(" ", "_").lower()
        box = item.get("box_2d") if isinstance(item, dict) else item
        labeled.append((label, box))

    # Group paired items by base, sort by center x, assign left then right
    by_base = {b: [] for b in PAIRED_BASES}
    unpaired = []
    for label, box in labeled:
        if label in PAIRED_LABELS and box and len(box) == 4:
            base = label.replace("_1", "").replace("_2", "")
            if base in by_base:
                by_base[base].append((label, box, _box_center_x(box)))
        else:
            unpaired.append((label, box))

    resolved = list(unpaired)
    for base in PAIRED_BASES:
        group = by_base[base]
        if len(group) != 2:
            resolved.extend((l, b) for l, b, _ in group)
            continue
        group.sort(key=lambda t: t[2])
        (l1, b1, cx1), (l2, b2, cx2) = group
        resolved.append(("left_" + base, b1))
        resolved.append(("right_" + base, b2))
        print(f"[bboxes] {base}: left cx={cx1:.0f}, right cx={cx2:.0f} (by x-position)")

    # Crop and save
    os.makedirs(outputFolder, exist_ok=True)
    for label, box in resolved:
        if box and len(box) == 4:
            abs_y1 = int(box[0]/1000 * height)
            abs_x1 = int(box[1]/1000 * width)
            abs_y2 = int(box[2]/1000 * height)
            abs_x2 = int(box[3]/1000 * width)

            crop = image.crop((abs_x1, abs_y1, abs_x2, abs_y2))
            save_path = os.path.join(outputFolder, f"{label}.png")
            crop.save(save_path)


def segmentation_masks(client, inputPath, outputFolder, parts=None, model: str = "gemini-3-flash-preview"):
    """
    Generates segmentation masks and transparent cutouts for parts of a figure.

    Args:
        client: The Google GenAI client instance.
        inputPath: Path to the input image file.
        outputFolder: Path where the masks and cutouts will be saved.
        parts: Optional list of part labels to detect.
        model: Model to use for segmentation.
    """
    temperature = 0

    if parts is None:
        parts = [
            "head",
            "torso",
            "right_upperarm",
            "left_upperarm",
            "right_forearm",
            "left_forearm",
            "right_hand",
            "left_hand",
            "right_thigh",
            "left_thigh",
            "right_calf",
            "left_calf",
            "right_foot",
            "left_foot",
        ]

    parts_list = ", ".join([f"'{part}'" for part in parts])

    prompt = f"""
    Task: Detect and return segmentation masks. Do not edit, redraw, or reinterpret the image.
    Detect exactly these parts of the figure: {parts_list}.
    Return a JSON array of objects, each with 'box_2d', 'mask', and 'label'.
    box_2d is [ymin, xmin, ymax, xmax] normalized to 0-1000.
    mask is a base64-encoded PNG (optionally prefixed with data:image/png;base64,).
    Keep the original drawing exactly as-is: same pose, same fingers, same proportions, same design. No alterations.
    """

    if not os.path.exists(inputPath):
        raise FileNotFoundError(f"Image file not found: {inputPath}")

    image = Image.open(inputPath)
    width, height = image.size

    response = client.models.generate_content(
        model=model,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=temperature,
        ),
    )

    segmentation = json.loads(parse_json(response.text))
    items = segmentation if isinstance(segmentation, list) else segmentation.get("items", [])

    os.makedirs(outputFolder, exist_ok=True)

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        label = item.get("label", f"part_{i}").replace(" ", "_").lower()
        box = item.get("box_2d")
        mask_str = item.get("mask")

        if not box or len(box) != 4 or not mask_str:
            continue

        abs_y1 = int(box[0] / 1000 * height)
        abs_x1 = int(box[1] / 1000 * width)
        abs_y2 = int(box[2] / 1000 * height)
        abs_x2 = int(box[3] / 1000 * width)

        if abs_y1 >= abs_y2 or abs_x1 >= abs_x2:
            continue

        if mask_str.startswith("data:image/png;base64,"):
            mask_str = mask_str.split(",", 1)[1]

        try:
            mask_data = base64.b64decode(mask_str)
        except (ValueError, TypeError):
            continue

        mask = Image.open(io.BytesIO(mask_data)).convert("L")
        mask = mask.resize((abs_x2 - abs_x1, abs_y2 - abs_y1), Image.Resampling.BILINEAR)
        mask = mask.point(lambda p: 255 if p > 128 else 0)

        crop = image.crop((abs_x1, abs_y1, abs_x2, abs_y2)).convert("RGBA")
        crop.putalpha(mask)

        mask_filename = f"{label}_{i}_mask.png"
        segment_filename = f"{label}_{i}_segment.png"

        mask.save(os.path.join(outputFolder, mask_filename))
        crop.save(os.path.join(outputFolder, segment_filename))
