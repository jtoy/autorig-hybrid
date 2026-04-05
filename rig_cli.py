#!/usr/bin/env python3
"""
rig_cli.py — Auto-rig CLI

Run the full pipeline (diecut → bboxes → background removal → rig) on a
single image or every image in a folder, producing a rig.json for each one
exactly as the UI does.

Usage examples
--------------
# Default: process every image in ./resources
python rig_cli.py

# Single image
python rig_cli.py tobyturtle.png

# Folder
python rig_cli.py resources/images/

# Custom output dir & model overrides
python rig_cli.py resources/ --output-dir out/ --diecut-rounds 3

# Print resulting JSON to stdout (in addition to saving files)
python rig_cli.py tobyturtle.png --print-json
"""

import argparse
import json
import os
import sys
import tempfile

import dotenv

dotenv.load_dotenv()

# Make sure the processing package is importable regardless of cwd.
sys.path.insert(0, os.path.dirname(__file__))

from google import genai

from processing import diecut, bboxes, rig
from processing.simple_background import remove_background_simple
from processing.genai_background import remove_background_genai

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

PART_NAMES = [
    "head",
    "torso",
    "left_upperarm",
    "right_upperarm",
    "left_forearm",
    "right_forearm",
    "left_hand",
    "right_hand",
    "left_thigh",
    "right_thigh",
    "left_calf",
    "right_calf",
    "left_foot",
    "right_foot",
]


def _collect_images(input_path: str) -> list[str]:
    """Return a list of image paths from a file or directory."""
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        return [input_path]

    if os.path.isdir(input_path):
        images = sorted(
            os.path.join(input_path, name)
            for name in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, name))
            and os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS
        )
        if not images:
            raise FileNotFoundError(
                f"No supported images found in '{input_path}'."
            )
        return images

    raise FileNotFoundError(f"'{input_path}' is not a file or directory.")


def _remove_backgrounds(
    parts_dir: str,
    use_genai: bool,
    tolerance: int,
    client,
) -> None:
    """Strip backgrounds from all part PNGs in-place."""
    for name in PART_NAMES:
        path = os.path.join(parts_dir, f"{name}.png")
        if not os.path.exists(path):
            continue
        if use_genai:
            remove_background_genai(client, path, tolerance=tolerance)
        else:
            remove_background_simple(path, tolerance)


def run_pipeline(
    client,
    image_path: str,
    work_dir: str,
    diecut_rounds: int = 5,
    diecut_model: str | None = None,
    vision_model: str | None = None,
    rig_model: str | None = None,
    use_genai_background: bool = False,
    tolerance: int = 30,
    fail_on_review: bool = False,
) -> dict:
    """
    Full pipeline for one image — mirrors what the UI /api/run-all endpoint does.

    Returns the rig JSON dict (same payload the UI sends back to the browser).
    """
    os.makedirs(work_dir, exist_ok=True)

    diecut_path = os.path.join(work_dir, "diecut.png")
    parts_dir = os.path.join(work_dir, "parts")
    rig_path = os.path.join(work_dir, "rig.json")

    diecut_kwargs: dict = {}
    if diecut_model:
        diecut_kwargs["model"] = diecut_model
    if vision_model:
        diecut_kwargs["judge_model"] = vision_model

    bbox_kwargs: dict = {}
    if vision_model:
        bbox_kwargs["model"] = vision_model

    rig_kwargs: dict = {}
    if rig_model:
        rig_kwargs["model"] = rig_model

    # ── Step 1 / 4  Diecut ────────────────────────────────────────────────────
    print(f"  [1/4] diecut  →  {diecut_path}")
    diecut(
        client,
        image_path,
        diecut_path,
        fail_on_review,
        diecut_rounds,
        **diecut_kwargs,
    )

    # ── Step 2 / 4  Bounding-box crop ─────────────────────────────────────────
    print(f"  [2/4] bboxes  →  {parts_dir}/")
    os.makedirs(parts_dir, exist_ok=True)
    bboxes(client, diecut_path, parts_dir, **bbox_kwargs)

    # ── Step 3 / 4  Background removal ────────────────────────────────────────
    print(f"  [3/4] bg-removal  (genai={use_genai_background})")
    _remove_backgrounds(parts_dir, use_genai_background, tolerance, client)

    # ── Step 4 / 4  Rig generation ────────────────────────────────────────────
    print(f"  [4/4] rig  →  {rig_path}")
    rig(client, image_path, parts_dir, rig_path, **rig_kwargs)

    with open(rig_path, "r", encoding="utf-8") as f:
        rig_data = json.load(f)

    return rig_data


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rig_cli.py",
        description=(
            "Auto-rig pipeline: diecut → bboxes → bg-removal → rig. "
            "Accepts a single image or a folder (default: resources/)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="resources",
        metavar="IMAGE_OR_FOLDER",
        help="Image file or folder of images (default: resources/).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "Root output directory. Each image gets a sub-folder named after "
            "its stem. Defaults to a temp dir when processing a single image, "
            "or '<input_folder>/rig_output/' when processing a folder."
        ),
    )
    parser.add_argument(
        "--diecut-rounds",
        type=int,
        default=5,
        metavar="N",
        help="Number of diecut refinement rounds (default: 5).",
    )
    parser.add_argument(
        "--diecut-model",
        default=None,
        metavar="MODEL",
        help="Gemini model for diecut generation.",
    )
    parser.add_argument(
        "--vision-model",
        default=None,
        metavar="MODEL",
        help="Gemini model for diecut judging and bbox detection.",
    )
    parser.add_argument(
        "--rig-model",
        default=None,
        metavar="MODEL",
        help="Gemini model for rig parameter generation.",
    )
    parser.add_argument(
        "--use-genai-background",
        action="store_true",
        help="Use GenAI background removal instead of the fast local remover.",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=30,
        metavar="N",
        help="Color tolerance for local background removal (default: 30).",
    )
    parser.add_argument(
        "--fail-on-review",
        action="store_true",
        help="Abort if the diecut judge rejects the output.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print each rig JSON to stdout after saving.",
    )

    args = parser.parse_args()

    # ── Resolve images ─────────────────────────────────────────────────────────
    try:
        image_paths = _collect_images(args.input)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    # ── Resolve output root ────────────────────────────────────────────────────
    single_image = os.path.isfile(args.input)
    if args.output_dir:
        output_root = args.output_dir
    elif single_image:
        # Put outputs next to the source image in a sibling folder
        stem = os.path.splitext(os.path.basename(args.input))[0]
        output_root = os.path.join(os.path.dirname(os.path.abspath(args.input)), f"{stem}_rig")
    else:
        output_root = os.path.join(os.path.abspath(args.input), "rig_output")

    os.makedirs(output_root, exist_ok=True)

    # ── Run ────────────────────────────────────────────────────────────────────
    client = genai.Client()
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}

    total = len(image_paths)
    for idx, image_path in enumerate(image_paths, 1):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        work_dir = os.path.join(output_root, stem)
        print(f"\n[{idx}/{total}] {image_path}  →  {work_dir}")

        try:
            rig_data = run_pipeline(
                client=client,
                image_path=image_path,
                work_dir=work_dir,
                diecut_rounds=args.diecut_rounds,
                diecut_model=args.diecut_model,
                vision_model=args.vision_model,
                rig_model=args.rig_model,
                use_genai_background=args.use_genai_background,
                tolerance=args.tolerance,
                fail_on_review=args.fail_on_review,
            )
            results[stem] = rig_data
            rig_out = os.path.join(work_dir, "rig.json")
            print(f"  ✓  rig saved  →  {rig_out}")

            if args.print_json:
                print(json.dumps(rig_data, indent=2))

        except Exception as exc:
            errors[stem] = str(exc)
            print(f"  ✗  ERROR: {exc}", file=sys.stderr)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Done.  {len(results)}/{total} image(s) rigged successfully.")
    if errors:
        print(f"Errors ({len(errors)}):")
        for name, msg in errors.items():
            print(f"  {name}: {msg}")

    print(f"\nOutput directory: {output_root}")
    for stem in results:
        print(f"  {stem}/rig.json")

    # Exit with non-zero code if any image failed
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
