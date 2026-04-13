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

Usage
-----
    # Process all characters, with Gemini:
    python lasso_batch.py

    # Only tank, no Gemini (fast local-only run):
    python lasso_batch.py --no-gemini tank

    # Custom dirs:
    python lasso_batch.py --lasso-dir resources/lasso --output-dir outputs tank
"""

import argparse
import json
import os
import shutil
import sys

import dotenv

dotenv.load_dotenv()

from PIL import Image

from processing.refine_part import refine_part_with_models
from processing.simple_background import remove_background_simple

# Directories searched for the source image, in priority order.
_IMAGE_SEARCH_DIRS = ["resources/images", "resources/characters", "resources"]
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


# ── Image helpers ─────────────────────────────────────────────────────────────


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


# ── Gemini refinement (mirrors ui/server.py logic) ───────────────────────────


def _refine_with_gemini(
    lasso_path: str,
    full_image_path: str,
    body_part: str,
    output_path: str,
) -> bool:
    """
    Send the raw lasso crop + full character image to Gemini and ask it to
    produce a clean, completed body-part image with a white background.

    Returns True if an image was written to output_path, False otherwise.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("    [gemini] google-genai not installed — skipping refinement")
        return False

    prompt = (
        "This is ONE PART of a 2d rigged character.\n"
        f"This should represent a {body_part}.\n"
        "Do NOT include any background elements or other parts. "
        "Do NOT include any other parts or elements. "
        "Simply return the body part completed and ready to be used in a 2D rig "
        "with a white background. "
        "Return this body part COMPLETED and ready to be used in a 2D rig with a white background."
    )
    model = "gemini-3.1-flash-image-preview"
    print(f"    [gemini] model={model}  part={body_part!r}")

    client = genai.Client()
    lasso_img = Image.open(lasso_path).convert("RGB")
    full_img = Image.open(full_image_path).convert("RGB")

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(
                text=(
                    "Image 1 is the FULL original character.\n"
                    "Image 2 is the lasso-extracted rig part.\n"
                    + prompt
                )
            ),
            full_img,
            lasso_img,
        ],
        config=types.GenerateContentConfig(temperature=0),
    )

    for part in getattr(response, "parts", []) or []:
        if part.inline_data is not None:
            part.as_image().save(output_path)
            return True

    # Log any text the model returned instead of an image
    for part in getattr(response, "parts", []) or []:
        if getattr(part, "text", None):
            preview = part.text[:200]
            print(f"    [gemini] No image in response. text_preview={preview!r}")
            break

    return False


# ── Image lookup ──────────────────────────────────────────────────────────────


def _find_image(character: str, repo_root: str) -> str | None:
    """Return the first matching image for *character* across search dirs."""
    for rel_dir in _IMAGE_SEARCH_DIRS:
        for ext in _IMAGE_EXTENSIONS:
            candidate = os.path.join(repo_root, rel_dir, f"{character}{ext}")
            if os.path.isfile(candidate):
                return candidate
    return None


# ── Per-character processing ──────────────────────────────────────────────────


def _scale_polygon(polygon: list, scale_x: float, scale_y: float) -> list:
    return [[int(round(p[0] * scale_x)), int(round(p[1] * scale_y))] for p in polygon]


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
            print(f"[{character}] No *_lasso_coords.json found — skipping")
            return
        lasso_json = os.path.join(char_dir, candidates[0])

    with open(lasso_json, encoding="utf-8") as f:
        lasso_data = json.load(f)

    parts = lasso_data.get("parts", [])
    if not parts:
        print(f"[{character}] No parts in {os.path.basename(lasso_json)} — skipping")
        return

    # Locate source image
    image_path = _find_image(character, repo_root)
    # Also check inside the char_dir itself (user may have placed it there)
    if image_path is None:
        for ext in _IMAGE_EXTENSIONS:
            candidate = os.path.join(char_dir, f"{character}{ext}")
            if os.path.isfile(candidate):
                image_path = candidate
                break
    if image_path is None:
        print(f"[{character}] Could not find source image — skipping")
        return

    # Determine polygon scaling if image dimensions differ from JSON record
    with Image.open(image_path) as img:
        actual_w, actual_h = img.size
    lasso_w = lasso_data.get("image_width", actual_w)
    lasso_h = lasso_data.get("image_height", actual_h)
    scale_x = actual_w / lasso_w if lasso_w else 1.0
    scale_y = actual_h / lasso_h if lasso_h else 1.0
    needs_scale = abs(scale_x - 1.0) > 0.001 or abs(scale_y - 1.0) > 0.001
    if needs_scale:
        print(
            f"[{character}] Scaling polygons: lasso={lasso_w}×{lasso_h}  "
            f"actual={actual_w}×{actual_h}  scale=({scale_x:.3f}, {scale_y:.3f})"
        )

    # Prepare output directories
    char_out = os.path.join(output_dir, character)
    raw_dir = os.path.join(char_out, "raw")
    refined_dir = os.path.join(char_out, "refined")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(refined_dir, exist_ok=True)

    print(
        f"[{character}] image={os.path.relpath(image_path, repo_root)}  "
        f"parts={len(parts)}  gemini={'on' if use_gemini else 'off'}"
    )

    ok_count = 0
    for part_entry in parts:
        label = part_entry["label"]
        polygon = part_entry["polygon"]

        if needs_scale:
            polygon = _scale_polygon(polygon, scale_x, scale_y)

        raw_path = os.path.join(raw_dir, f"{label}.png")
        refined_path = os.path.join(refined_dir, f"{label}.png")

        print(f"  [{label}] lasso cut …")
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
            print(f"  [{label}] gemini refinement …")
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

    print(f"[{character}] done  {ok_count}/{len(parts)} part(s) saved\n")


# ── Entry point ───────────────────────────────────────────────────────────────


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
        help="Character folder names to process (default: all folders in --lasso-dir)",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    lasso_dir = os.path.join(repo_root, args.lasso_dir)
    output_dir = os.path.join(repo_root, args.output_dir)

    if not os.path.isdir(lasso_dir):
        sys.exit(f"Lasso directory not found: {lasso_dir}")

    characters = args.characters or [
        d for d in sorted(os.listdir(lasso_dir)) if os.path.isdir(os.path.join(lasso_dir, d))
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
