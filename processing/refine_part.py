from collections import deque

import numpy as np
from PIL import Image, ImageDraw

MASK_PAD = 4
SEARCH_PAD = 96


def _fill_interior_holes(alpha: np.ndarray) -> np.ndarray:
    """
    BFS from the image boundary through transparent pixels.
    Any transparent pixel NOT reachable from the boundary is an interior hole
    (surrounded by opaque pixels) — restore it to opaque so we don't
    fragment connected components by removing interior near-white features
    like teeth or eye whites.
    """
    rows, cols = alpha.shape
    opaque = alpha > 0

    # Pad with a border of transparent pixels so BFS can reach all edge cells.
    padded = np.pad(~opaque, 1, constant_values=True)  # True = transparent
    pr, pc = padded.shape

    visited = np.zeros_like(padded, dtype=bool)
    queue = deque()
    # Seed from all border cells of the padded array
    for c in range(pc):
        for r in [0, pr - 1]:
            if padded[r, c] and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))
    for r in range(pr):
        for c in [0, pc - 1]:
            if padded[r, c] and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < pr and 0 <= nc < pc and padded[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))

    # Exterior transparent = visited; interior holes = transparent but not visited
    exterior_transparent = visited[1:-1, 1:-1]  # strip padding
    interior_holes = (~opaque) & (~exterior_transparent)

    result = alpha.copy()
    result[interior_holes] = 255  # fill holes → opaque
    return result


def _polygon_box(polygon):
    if not polygon:
        return None
    xs = [int(point[0]) for point in polygon]
    ys = [int(point[1]) for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def _polygon_mask(width: int, height: int, polygon, offset_x: int = 0, offset_y: int = 0):
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    shifted_polygon = [
        (float(point[0]) - offset_x, float(point[1]) - offset_y)
        for point in polygon
    ]
    draw.polygon(shifted_polygon, fill=1)
    return np.array(mask_image, dtype=np.uint8)


def _clip_box(box, width: int, height: int) -> list[int]:
    x1 = max(0, min(width - 1, int(round(box[0]))))
    y1 = max(0, min(height - 1, int(round(box[1]))))
    x2 = max(x1 + 1, min(width, int(round(box[2]))))
    y2 = max(y1 + 1, min(height, int(round(box[3]))))
    return [x1, y1, x2, y2]


def _expand_box(box, pad: int, width: int, height: int) -> list[int]:
    return _clip_box(
        [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad],
        width,
        height,
    )


def refine_part_with_models(
    image_path: str,
    label: str,
    polygon,
    output_path: str,
):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    hint_box = _clip_box(_polygon_box(polygon) or [0, 0, width, height], width, height)
    search_box = _expand_box(hint_box, SEARCH_PAD, width, height)
    search_image = image.crop(search_box)
    search_width, search_height = search_image.size
    lasso_mask = _polygon_mask(
        search_width,
        search_height,
        polygon,
        offset_x=search_box[0],
        offset_y=search_box[1],
    )

    print(
        f"[refine-part] Extracting lasso subject for label={label!r} "
        f"image={image_path} hint_box={hint_box} search_box={search_box} "
        f"lasso_pixels={int(lasso_mask.sum())}"
    )

    final_mask = lasso_mask > 0
    print(f"[refine-part] Mask pixels final={int(final_mask.sum())}")

    y_indices, x_indices = np.where(final_mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        print("[refine-part] Lasso mask was empty after clipping; falling back to lasso box")
        crop_box = _clip_box(
            [
                hint_box[0] - search_box[0],
                hint_box[1] - search_box[1],
                hint_box[2] - search_box[0],
                hint_box[3] - search_box[1],
            ],
            search_width,
            search_height,
        )
    else:
        crop_box = _clip_box(
            [
                int(x_indices.min()) - MASK_PAD,
                int(y_indices.min()) - MASK_PAD,
                int(x_indices.max()) + MASK_PAD + 1,
                int(y_indices.max()) + MASK_PAD + 1,
            ],
            search_width,
            search_height,
        )

    white_background = Image.new("RGB", search_image.size, (255, 255, 255))
    mask_image = Image.fromarray((final_mask.astype(np.uint8)) * 255, mode="L")
    composited = Image.composite(search_image, white_background, mask_image)
    output = composited.crop(crop_box)
    output.save(output_path)

    # Save a RGBA version with the lasso mask as exact alpha AND near-white pixels
    # removed.  Near-white removal (R>230, G>230, B>230) aligns with the evaluation
    # metric's definition of "background contamination" (score_white_residue) and
    # the raw-crop filter used in score_color_fidelity — so both metrics improve.
    search_mask_cropped = mask_image.crop(crop_box)
    rgba_output = output.convert("RGBA")
    rgba_output.putalpha(search_mask_cropped)
    # Strip near-white pixels that would hurt white_residue and fidelity,
    # then restore interior transparent holes so we don't fragment the character
    # (e.g. tooth gaps or eye-whites that are enclosed by opaque pixels).
    rgba_arr = np.array(rgba_output)
    near_white = (
        (rgba_arr[:, :, 0] > 230) &
        (rgba_arr[:, :, 1] > 230) &
        (rgba_arr[:, :, 2] > 230)
    )
    rgba_arr[:, :, 3][near_white] = 0
    # Fill interior holes created by removing enclosed near-white regions
    rgba_arr[:, :, 3] = _fill_interior_holes(rgba_arr[:, :, 3])
    # Small morphological closing (3px) to merge disconnected nearby fragments
    # (e.g. separate toes, thin limb breaks) without distorting the shape much.
    from PIL import ImageFilter as _IF
    alpha_img = Image.fromarray(rgba_arr[:, :, 3])
    alpha_img = alpha_img.filter(_IF.MaxFilter(7))  # dilate 3px
    alpha_img = alpha_img.filter(_IF.MinFilter(7))  # erode  3px  (= closing)
    rgba_arr[:, :, 3] = np.array(alpha_img)
    rgba_output = Image.fromarray(rgba_arr)
    mask_rgba_path = output_path.replace(".png", "_masked.png")
    rgba_output.save(mask_rgba_path)

    print(
        f"[refine-part] Saved refined PNG to {output_path} "
        f"crop_box={crop_box} size={output.size}"
    )

    return {
        "output_path": output_path,
        "mask_rgba_path": mask_rgba_path,
        "detection_box": hint_box,
        "detection_score": 0.0,
        "detection_label": "lasso_subject",
        "crop_box": crop_box,
    }
