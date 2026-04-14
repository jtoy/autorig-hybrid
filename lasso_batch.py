"""
Batch lasso-cut pipeline from saved selection coordinates.

For every character folder under resources/lasso/ that contains a
*_lasso_coords.json, this script:

  1. Locates the matching source image (resources/images/, resources/characters/,
     or the same folder as the JSON).
  2. Scales polygon coordinates if the source image dimensions differ from those
     recorded in the JSON (allows re-running on different-resolution exports).
  3. For each part polygon, runs the lasso mask crop via
     processing.refine_part.refine_part_with_models.
  4. Optionally calls Gemini to complete/refine the crop
     (disable with --no-gemini).
  5. Removes background and trims transparent padding.
  6. Saves both the raw crop and the final refined PNG to
     outputs/<character>/raw/<label>.png and outputs/<character>/refined/<label>.png.
  7. Builds a three-panel composite image:
       - Original character with lasso polygon overlays
       - Raw lasso crops arranged in a grid
       - AI-refined crops in the same grid layout
     Saved to outputs/<character>/<character>_composite.png.

Usage
-----
    # Process all characters found under resources/lasso/, with Gemini:
    python lasso_batch.py

    # Only process tank:
    python lasso_batch.py tank

    # Only tank, no Gemini (fast local-only run):
    python lasso_batch.py --no-gemini tank

    # Custom dirs:
    python lasso_batch.py --lasso-dir resources/lasso --output-dir outputs tank
"""

import argparse
import json
import math
import os
import shutil
import sys

import dotenv

dotenv.load_dotenv()

from PIL import Image, ImageDraw, ImageFont

from processing.refine_part import refine_part_with_models
from processing.simple_background import remove_background_simple

# ── Constants ─────────────────────────────────────────────────────────────────

# Directories searched for the source image, in priority order.
_IMAGE_SEARCH_DIRS = ["resources/images", "resources/characters", "resources"]
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

# ── Composite layout constants ─────────────────────────────────────────────────
# Colours match the lasso UI theme.
_BG       = (13,  13,  26)    # --bg
_SURFACE  = (22,  22,  39)    # --surface
_SURFACE2 = (28,  28,  48)    # --surface2
_BORDER   = (42,  42,  72)    # --border
_ACCENT   = (124, 111, 255)   # --accent
_TEXT     = (226, 226, 240)   # --text
_TEXT_DIM = (119, 120, 152)   # --text-dim
_SUCCESS  = (74,  222, 128)   # --success

# Cycling colours for lasso overlays on the original image.
_OVERLAY_COLORS = [
    (255, 122,  80, 90),   # orange
    (124, 111, 255, 90),   # purple
    ( 74, 222, 128, 90),   # green
    (251, 191,  36, 90),   # yellow
    (248, 113, 113, 90),   # red
    ( 56, 189, 248, 90),   # cyan
    (244, 114, 182, 90),   # pink
    (167, 243, 208, 90),   # mint
]
_OUTLINE_COLORS = [
    (255, 122,  80),
    (124, 111, 255),
    ( 74, 222, 128),
    (251, 191,  36),
    (248, 113, 113),
    ( 56, 189, 248),
    (244, 114, 182),
    (167, 243, 208),
]

_PANEL_W  = 400   # width of each of the three panels
_COLS     = 2     # grid columns in the parts panels
_PAD      = 10    # general padding
_HEADER_H = 44    # height of each panel header
_TITLE_H  = 40    # height of the top title bar
_CELL_IMG = 160   # image area height/width inside each grid cell
_CELL_LBL = 18    # label row height below each cell image
_CELL_H   = _CELL_IMG + _CELL_LBL + _PAD   # total cell height (img + label + gap)
_CELL_W   = (_PANEL_W - _PAD * (_COLS + 1)) // _COLS   # cell width derived from panel width


# ── Image helpers ──────────────────────────────────────────────────────────────


def _trim_transparent(image_path: str, alpha_threshold: int = 20) -> None:
    image = Image.open(image_path).convert("RGBA")
    alpha = image.split()[3]
    alpha = alpha.point(lambda v: 0 if v < alpha_threshold else v)
    bbox = alpha.getbbox()
    if bbox:
        image.crop(bbox).save(image_path)
    else:
        image.save(image_path)


def _cleanup_part(image_path: str, tolerance: int = 30) -> None:
    """Remove white/corner background then trim empty alpha border."""
    remove_background_simple(image_path, tolerance)
    _trim_transparent(image_path)


def _fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Resize to fit within (max_w, max_h), preserving aspect ratio."""
    img = img.convert("RGBA")
    if img.width == 0 or img.height == 0:
        return img
    ratio = min(max_w / img.width, max_h / img.height, 1.0)
    new_w = max(1, round(img.width  * ratio))
    new_h = max(1, round(img.height * ratio))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ["segoeuil.ttf", "segoeui.ttf", "arial.ttf", "Arial.ttf",
                 "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ── Gemini refinement ──────────────────────────────────────────────────────────


def _refine_with_gemini(
    lasso_path: str,
    full_image_path: str,
    body_part: str,
    output_path: str,
) -> bool:
    """
    Send the raw lasso crop + full character image to Gemini and ask it to
    produce a clean, completed body-part with a white background.
    Returns True if an image was written to output_path.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("    [gemini] google-genai not installed -- skipping refinement")
        return False

    prompt = (
        f"Image 1: a {body_part} crop from a 2D character. Image 2: the full character for context.\n"
        "Return Image 1 EXACTLY as-is but with the white background cleanly removed, "
        "placed on a fresh white background. "
        "Do not change the content, colors, or style of the body part at all."
    )
    model = "gemini-3.1-flash-image-preview"
    print(f"    [gemini] model={model}  part={body_part!r}")

    client = genai.Client()
    lasso_img = Image.open(lasso_path).convert("RGB")
    full_img  = Image.open(full_image_path).convert("RGB")

    # Selective upscaling: only upscale where Gemini has too little signal.
    # - Very small (< 100px short side): scale up to 512px — hands, tiny extremities.
    # - Medium (100-255px short side): gentle 2× — head, feet; thighs stay at native.
    # - Large (>= 256px): no upscale — already enough detail.
    w, h = lasso_img.size
    short_side = min(w, h)
    if short_side < 100:
        scale = 512 / short_side
        new_w, new_h = round(w * scale), round(h * scale)
        lasso_img = lasso_img.resize((new_w, new_h), Image.LANCZOS)
        print(f"    [gemini] upscaled crop {w}x{h} -> {new_w}x{new_h} ({scale:.1f}x tiny)")
    elif short_side < 256:
        new_w, new_h = w * 2, h * 2
        lasso_img = lasso_img.resize((new_w, new_h), Image.LANCZOS)
        print(f"    [gemini] upscaled crop {w}x{h} -> {new_w}x{new_h} (2x medium)")

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=prompt),
            lasso_img,
            full_img,
        ],
        config=types.GenerateContentConfig(temperature=0),
    )

    for part in getattr(response, "parts", []) or []:
        if part.inline_data is not None:
            part.as_image().save(output_path)
            return True

    for part in getattr(response, "parts", []) or []:
        if getattr(part, "text", None):
            print(f"    [gemini] no image returned. preview={part.text[:200]!r}")
            break

    return False


# ── Source image lookup ────────────────────────────────────────────────────────


def _find_image(character: str, repo_root: str) -> str | None:
    for rel_dir in _IMAGE_SEARCH_DIRS:
        for ext in _IMAGE_EXTENSIONS:
            candidate = os.path.join(repo_root, rel_dir, f"{character}{ext}")
            if os.path.isfile(candidate):
                return candidate
    return None


def _scale_polygon(polygon: list, scale_x: float, scale_y: float) -> list:
    return [[round(p[0] * scale_x), round(p[1] * scale_y)] for p in polygon]


# ── Composite image builder ────────────────────────────────────────────────────


def _draw_header_bar(
    draw: ImageDraw.Draw,
    x: int, y: int, w: int, h: int,
    title: str,
    color: tuple,
    font: ImageFont.ImageFont,
) -> None:
    draw.rectangle([x, y, x + w - 1, y + h - 1], fill=_SURFACE)
    draw.line([(x, y + h - 1), (x + w - 1, y + h - 1)], fill=_BORDER)
    draw.text((x + _PAD, y + h // 2), title, fill=color, font=font, anchor="lm")


def _draw_parts_panel(
    canvas: Image.Image,
    draw: ImageDraw.Draw,
    parts: list,          # [(label, Image)]
    px: int, py: int,     # top-left of the content area (below header)
    font_lbl: ImageFont.ImageFont,
) -> None:
    """Tile part images in a 2-column grid inside the content area."""
    if not parts:
        draw.text(
            (px + _PANEL_W // 2, py + _CELL_H),
            "no parts",
            fill=_TEXT_DIM, font=font_lbl, anchor="mm",
        )
        return

    for i, (label, img) in enumerate(parts):
        col = i % _COLS
        row = i // _COLS
        cx = px + _PAD + col * (_CELL_W + _PAD)
        cy = py + _PAD + row * _CELL_H

        # Cell background + border
        draw.rectangle([cx, cy, cx + _CELL_W - 1, cy + _CELL_IMG - 1],
                       fill=_SURFACE2, outline=_BORDER, width=1)

        # Image centred in the cell
        fitted = _fit_image(img, _CELL_W - 4, _CELL_IMG - 4)
        ox = cx + (_CELL_W   - fitted.width)  // 2
        oy = cy + (_CELL_IMG  - fitted.height) // 2
        canvas.paste(fitted, (ox, oy), fitted)

        # Label below
        display = label.replace("_", " ")
        draw.text(
            (cx + _CELL_W // 2, cy + _CELL_IMG + _CELL_LBL // 2),
            display,
            fill=_TEXT_DIM, font=font_lbl, anchor="mm",
        )


def _draw_original_panel(
    canvas: Image.Image,
    draw: ImageDraw.Draw,
    orig: Image.Image,
    px: int, py: int,          # content area top-left
    content_h: int,
    scaled_polys: list,        # [(label, scaled_polygon)]
    font_lbl: ImageFont.ImageFont,
) -> None:
    """Draw the original image scaled to fit, with lasso polygon overlays."""
    max_w = _PANEL_W - _PAD * 2
    max_h = content_h - _PAD * 2

    scale = min(max_w / orig.width, max_h / orig.height, 1.0)
    disp_w = round(orig.width  * scale)
    disp_h = round(orig.height * scale)
    ox = px + (_PANEL_W - disp_w) // 2
    oy = py + (content_h - disp_h) // 2

    fitted = orig.convert("RGBA").resize((disp_w, disp_h), Image.LANCZOS)
    canvas.paste(fitted, (ox, oy), fitted)

    # Draw polygon overlays on a transparent layer, then composite
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)

    for idx, (label, poly) in enumerate(scaled_polys):
        color_idx = idx % len(_OVERLAY_COLORS)
        fill    = _OVERLAY_COLORS[color_idx]
        outline = _OUTLINE_COLORS[color_idx]

        # Map polygon from original image space → display space
        pts = [
            (ox + round(p[0] * scale), oy + round(p[1] * scale))
            for p in poly
        ]
        if len(pts) >= 3:
            ov_draw.polygon(pts, fill=fill)
            ov_draw.line(pts + [pts[0]], fill=outline + (220,), width=2)

        # Label at centroid
        cx = sum(p[0] for p in pts) // len(pts)
        cy = sum(p[1] for p in pts) // len(pts)
        ov_draw.text((cx, cy), label.replace("_", " "),
                     fill=(*outline, 255), font=font_lbl, anchor="mm")

    canvas.alpha_composite(overlay)


def build_composite(
    character: str,
    image_path: str,
    raw_dir: str,
    refined_dir: str,
    lasso_parts: list,   # [{"label": str, "polygon": [[x,y],...]}]
    scale_x: float,
    scale_y: float,
    output_path: str,
) -> None:
    """Render the three-panel composite and save to output_path."""
    n_parts = len(lasso_parts)
    n_rows  = max(1, math.ceil(n_parts / _COLS))
    content_h = _PAD + n_rows * _CELL_H + _PAD

    panel_h   = _HEADER_H + content_h
    total_w   = _PANEL_W * 3 + _PAD * 4
    total_h   = _TITLE_H + panel_h + _PAD

    canvas = Image.new("RGBA", (total_w, total_h), _BG)
    draw   = ImageDraw.Draw(canvas)

    font_title  = _load_font(15)
    font_header = _load_font(11)
    font_lbl    = _load_font(10)

    # ── Top title bar ──────────────────────────────────────────────────────────
    draw.rectangle([0, 0, total_w - 1, _TITLE_H - 1], fill=_SURFACE)
    draw.line([(0, _TITLE_H - 1), (total_w - 1, _TITLE_H - 1)], fill=_BORDER)
    draw.text(
        (_PAD * 2, _TITLE_H // 2),
        f"Auto-Rig Lasso  /  {character}",
        fill=_ACCENT, font=font_title, anchor="lm",
    )
    draw.text(
        (total_w - _PAD * 2, _TITLE_H // 2),
        f"{n_parts} part(s)",
        fill=_TEXT_DIM, font=font_lbl, anchor="rm",
    )

    # ── Panel positions ────────────────────────────────────────────────────────
    pxs = [_PAD + i * (_PANEL_W + _PAD) for i in range(3)]
    panel_y = _TITLE_H

    panel_meta = [
        ("Original + Selections", _ACCENT),
        ("Raw Lasso Cuts",        _TEXT_DIM),
        ("AI Refined",           _SUCCESS),
    ]
    for i, (title, color) in enumerate(panel_meta):
        _draw_header_bar(draw, pxs[i], panel_y, _PANEL_W, _HEADER_H, title, color, font_header)
        draw.rectangle(
            [pxs[i], panel_y, pxs[i] + _PANEL_W - 1, panel_y + panel_h - 1],
            outline=_BORDER, width=1,
        )

    content_y = panel_y + _HEADER_H

    # ── Panel 0: original + polygon overlays ──────────────────────────────────
    orig = Image.open(image_path)
    scaled_polys = [
        (
            p["label"],
            _scale_polygon(p["polygon"], scale_x, scale_y) if (abs(scale_x - 1) > 0.001 or abs(scale_y - 1) > 0.001) else p["polygon"],
        )
        for p in lasso_parts
    ]
    _draw_original_panel(canvas, draw, orig, pxs[0], content_y, content_h, scaled_polys, font_lbl)

    # ── Panels 1 & 2: part grids ──────────────────────────────────────────────
    def load_panel_parts(source_dir: str) -> list:
        result = []
        for p in lasso_parts:
            label = p["label"]
            path  = os.path.join(source_dir, f"{label}.png")
            if os.path.isfile(path):
                result.append((label, Image.open(path)))
        return result

    _draw_parts_panel(canvas, draw, load_panel_parts(raw_dir),     pxs[1], content_y, font_lbl)
    _draw_parts_panel(canvas, draw, load_panel_parts(refined_dir), pxs[2], content_y, font_lbl)

    canvas.convert("RGB").save(output_path, quality=95)
    print(f"[{character}] composite -> {os.path.relpath(output_path)}")


# ── Per-character processing ───────────────────────────────────────────────────


def process_character(
    character: str,
    lasso_dir: str,
    output_dir: str,
    repo_root: str,
    use_gemini: bool,
) -> None:
    char_dir = os.path.join(lasso_dir, character)
    if not os.path.isdir(char_dir):
        print(f"[{character}] Directory not found: {char_dir}")
        return

    # Find the lasso coords JSON
    lasso_json = os.path.join(char_dir, f"{character}_lasso_coords.json")
    if not os.path.isfile(lasso_json):
        candidates = [f for f in os.listdir(char_dir) if f.endswith("_lasso_coords.json")]
        if not candidates:
            print(f"[{character}] No *_lasso_coords.json found -- skipping")
            return
        lasso_json = os.path.join(char_dir, candidates[0])

    with open(lasso_json, encoding="utf-8") as f:
        lasso_data = json.load(f)

    lasso_parts = lasso_data.get("parts", [])
    if not lasso_parts:
        print(f"[{character}] No parts in {os.path.basename(lasso_json)} -- skipping")
        return

    # Locate source image
    image_path = _find_image(character, repo_root)
    if image_path is None:
        for ext in _IMAGE_EXTENSIONS:
            candidate = os.path.join(char_dir, f"{character}{ext}")
            if os.path.isfile(candidate):
                image_path = candidate
                break
    if image_path is None:
        print(f"[{character}] Could not find source image -- skipping")
        return

    # Polygon scaling if image size differs from JSON record
    with Image.open(image_path) as img:
        actual_w, actual_h = img.size
    lasso_w = lasso_data.get("image_width",  actual_w)
    lasso_h = lasso_data.get("image_height", actual_h)
    scale_x = actual_w / lasso_w if lasso_w else 1.0
    scale_y = actual_h / lasso_h if lasso_h else 1.0
    needs_scale = abs(scale_x - 1.0) > 0.001 or abs(scale_y - 1.0) > 0.001
    if needs_scale:
        print(
            f"[{character}] Scaling polygons: lasso={lasso_w}x{lasso_h}  "
            f"actual={actual_w}x{actual_h}  scale=({scale_x:.3f}, {scale_y:.3f})"
        )

    # Prepare output directories
    char_out    = os.path.join(output_dir, character)
    raw_dir     = os.path.join(char_out, "raw")
    refined_dir = os.path.join(char_out, "refined")
    os.makedirs(raw_dir,     exist_ok=True)
    os.makedirs(refined_dir, exist_ok=True)

    print(
        f"[{character}] image={os.path.relpath(image_path, repo_root)}  "
        f"parts={len(lasso_parts)}  gemini={'on' if use_gemini else 'off'}"
    )

    ok_count = 0
    for part_entry in lasso_parts:
        label   = part_entry["label"]
        polygon = part_entry["polygon"]

        if needs_scale:
            polygon = _scale_polygon(polygon, scale_x, scale_y)

        raw_path     = os.path.join(raw_dir,     f"{label}.png")
        refined_path = os.path.join(refined_dir, f"{label}.png")

        print(f"  [{label}] lasso cut ...")
        try:
            refine_part_with_models(
                image_path=image_path,
                label=label,
                polygon=polygon,
                output_path=raw_path,
            )
        except Exception as exc:
            print(f"  [{label}] ERROR during lasso cut: {exc}")
            continue

        if use_gemini:
            print(f"  [{label}] gemini refinement ...")
            try:
                success = _refine_with_gemini(
                    lasso_path=raw_path,
                    full_image_path=image_path,
                    body_part=label,
                    output_path=refined_path,
                )
                if success:
                    _cleanup_part(refined_path)
                    print(f"  [{label}] OK refined -> {os.path.relpath(refined_path, repo_root)}")
                else:
                    shutil.copy(raw_path, refined_path)
                    _cleanup_part(refined_path)
                    print(f"  [{label}] gemini returned no image; using raw crop")
            except Exception as exc:
                print(f"  [{label}] ERROR during gemini refinement: {exc}")
                shutil.copy(raw_path, refined_path)
        else:
            shutil.copy(raw_path, refined_path)
            print(f"  [{label}] OK raw crop -> {os.path.relpath(raw_path, repo_root)}")

        ok_count += 1

    # ── Composite image ────────────────────────────────────────────────────────
    composite_path = os.path.join(char_out, f"{character}_composite.png")
    try:
        build_composite(
            character=character,
            image_path=image_path,
            raw_dir=raw_dir,
            refined_dir=refined_dir,
            lasso_parts=lasso_parts,
            scale_x=scale_x,
            scale_y=scale_y,
            output_path=composite_path,
        )
    except Exception as exc:
        print(f"[{character}] ERROR building composite: {exc}")

    print(f"[{character}] done  {ok_count}/{len(lasso_parts)} part(s) saved\n")


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch lasso-cut pipeline from saved selection coordinates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--lasso-dir",
        default="resources/lasso",
        metavar="DIR",
        help="Root dir containing character sub-folders with lasso JSON files "
        "(default: resources/lasso)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        metavar="DIR",
        help="Root dir for output files (default: outputs/)",
    )
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Skip Gemini refinement and save the raw polygon crop only",
    )
    parser.add_argument(
        "characters",
        nargs="*",
        metavar="CHARACTER",
        help=(
            "Character folder names to process. "
            "If omitted, runs every character folder found in --lasso-dir."
        ),
    )
    args = parser.parse_args()

    repo_root  = os.path.dirname(os.path.abspath(__file__))
    lasso_dir  = os.path.join(repo_root, args.lasso_dir)
    output_dir = os.path.join(repo_root, args.output_dir)

    if not os.path.isdir(lasso_dir):
        sys.exit(f"Lasso directory not found: {lasso_dir}")

    characters = args.characters or [
        d for d in sorted(os.listdir(lasso_dir))
        if os.path.isdir(os.path.join(lasso_dir, d))
    ]
    if not characters:
        sys.exit(f"No character folders found in {lasso_dir}")

    use_gemini = not args.no_gemini

    print(f"Characters : {', '.join(characters)}")
    print(f"Output dir : {output_dir}")
    print(f"Gemini     : {'enabled' if use_gemini else 'disabled (--no-gemini)'}")
    print()

    for character in characters:
        process_character(
            character=character,
            lasso_dir=lasso_dir,
            output_dir=output_dir,
            repo_root=repo_root,
            use_gemini=use_gemini,
        )


if __name__ == "__main__":
    main()
