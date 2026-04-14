"""
Evaluation script for lasso_batch.py outputs.
Fixed — do not modify.

Scores the refined PNGs in outputs/<character>/refined/ against 5 metrics:

  1. connected_components  — 1 blob = perfect; more blobs = fragmented cut
  2. interior_holes        — no interior transparent holes = perfect
  3. area_ratio            — output pixel area vs lasso polygon area (scale check)
  4. white_residue         — non-transparent near-white pixels = bg contamination
  5. color_fidelity        — color histogram match between raw lasso and refined output
                             (low = Gemini hallucinated or extended with wrong content)

Each metric is normalized to [0, 1] (higher = better).
Final score is the mean across all metrics and all parts (higher = better).

Usage:
    python evaluate.py              # score all characters with outputs in outputs/
    python evaluate.py tank         # score only tank
    python evaluate.py --raw tank   # score raw crops instead of refined
"""

import argparse
import json
import os
import sys
from collections import deque

import numpy as np
from PIL import Image


# ── Polygon helpers ────────────────────────────────────────────────────────────


def _polygon_area_px(polygon: list) -> float:
    """Shoelace formula — returns area in pixels²."""
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


# ── Alpha mask helpers ─────────────────────────────────────────────────────────


def _alpha_binary(img_rgba: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Return bool mask: True where pixel is non-transparent."""
    return img_rgba[:, :, 3] > threshold


def _bfs_fill(binary: np.ndarray, seeds: list) -> np.ndarray:
    """
    BFS flood fill from seed pixels on a boolean grid.
    Returns a boolean mask of all pixels reachable from seeds
    through True values.
    """
    visited = np.zeros_like(binary, dtype=bool)
    rows, cols = binary.shape
    queue = deque()
    for r, c in seeds:
        if 0 <= r < rows and 0 <= c < cols and binary[r, c] and not visited[r, c]:
            visited[r, c] = True
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and binary[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))
    return visited


# ── Metric 1: Connected components ────────────────────────────────────────────


def _count_components(binary: np.ndarray) -> int:
    """Count 4-connected foreground components using BFS."""
    visited = np.zeros_like(binary, dtype=bool)
    rows, cols = binary.shape
    count = 0
    queue = deque()
    for start_r in range(rows):
        for start_c in range(cols):
            if binary[start_r, start_c] and not visited[start_r, start_c]:
                count += 1
                visited[start_r, start_c] = True
                queue.append((start_r, start_c))
                while queue:
                    r, c = queue.popleft()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and binary[nr, nc]
                            and not visited[nr, nc]
                        ):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
    return count


def score_connected_components(alpha: np.ndarray) -> float:
    """
    1.0  → single clean blob (perfect)
    0.5  → 2 blobs
    1/n  → n blobs
    0.0  → empty mask
    """
    binary = _alpha_binary(alpha)
    fg_pixels = binary.sum()
    if fg_pixels == 0:
        return 0.0

    # Ignore tiny noise blobs (< 0.1% of the foreground area)
    noise_threshold = max(1, int(fg_pixels * 0.001))
    visited = np.zeros_like(binary, dtype=bool)
    rows, cols = binary.shape
    significant = 0
    queue = deque()
    for start_r in range(rows):
        for start_c in range(cols):
            if binary[start_r, start_c] and not visited[start_r, start_c]:
                component = []
                visited[start_r, start_c] = True
                queue.append((start_r, start_c))
                while queue:
                    r, c = queue.popleft()
                    component.append((r, c))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and binary[nr, nc]
                            and not visited[nr, nc]
                        ):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                if len(component) > noise_threshold:
                    significant += 1

    return 1.0 / max(1, significant)


# ── Metric 2: Interior holes ───────────────────────────────────────────────────


def score_interior_holes(alpha: np.ndarray) -> float:
    """
    1.0  → no interior holes (occluded areas completed, or none needed)
    0.0  → hole area equals foreground area (entirely hollow)

    Algorithm:
      - Pad the alpha mask with a border of transparent pixels.
      - BFS-flood the transparent region from all padded border cells.
      - Any transparent pixel NOT reached is an interior hole.
    """
    binary = _alpha_binary(alpha)
    fg_pixels = int(binary.sum())
    if fg_pixels == 0:
        return 0.0

    # Pad with a 1-pixel transparent border so BFS can reach all exterior bg
    padded = np.pad(binary, pad_width=1, mode="constant", constant_values=False)
    rows, cols = padded.shape

    # BFS over the transparent (False) region starting from all border cells
    transparent_padded = ~padded
    seeds = []
    for c in range(cols):
        seeds.append((0, c))
        seeds.append((rows - 1, c))
    for r in range(rows):
        seeds.append((r, 0))
        seeds.append((r, cols - 1))

    exterior = _bfs_fill(transparent_padded, seeds)

    # Interior holes: transparent in padded but not reached by exterior BFS
    interior_holes = transparent_padded & ~exterior
    # Remove the padding border
    interior_holes = interior_holes[1:-1, 1:-1]

    hole_pixels = int(interior_holes.sum())
    hole_ratio = hole_pixels / fg_pixels
    return max(0.0, 1.0 - hole_ratio)


# ── Metric 3: Area ratio ───────────────────────────────────────────────────────


def score_area_ratio(alpha: np.ndarray, polygon_area: float) -> float:
    """
    1.0  → output pixel count matches polygon area exactly
    Degrades smoothly on a log scale — being 2× too big is as bad as 2× too small.
    Allows up to ~20% larger (for completed occluded areas) before penalising.

    Returns 0.0 if polygon_area is zero or the mask is empty.
    """
    if polygon_area <= 0:
        return 0.0
    output_px = int(_alpha_binary(alpha).sum())
    if output_px == 0:
        return 0.0
    ratio = output_px / polygon_area
    # Log-scale distance: 0 at ratio=1, 0.693 at ratio=0.5 or 2.0
    log_dist = abs(np.log(ratio))
    # Score: 1.0 at perfect, ~0 at 4× off
    return float(max(0.0, 1.0 - log_dist / np.log(4)))


# ── Metric 4: White residue ────────────────────────────────────────────────────


def score_white_residue(img_rgba: np.ndarray) -> float:
    """
    1.0  → no near-white contamination inside the foreground mask
    0.0  → all foreground pixels are near-white (pure bg leak)

    Near-white threshold: R > 230, G > 230, B > 230.
    """
    alpha = img_rgba[:, :, 3]
    rgb = img_rgba[:, :, :3]
    non_transparent = alpha > 127
    fg_count = int(non_transparent.sum())
    if fg_count == 0:
        return 0.0
    near_white = (rgb[:, :, 0] > 230) & (rgb[:, :, 1] > 230) & (rgb[:, :, 2] > 230)
    white_count = int((non_transparent & near_white).sum())
    white_ratio = white_count / fg_count
    # Amplify: 20% white is already very bad → score near 0
    return float(max(0.0, 1.0 - white_ratio * 5.0))


# ── Metric 5: Color fidelity (raw vs refined) ─────────────────────────────────


def score_color_fidelity(raw_path: str, refined_path: str, bins: int = 16) -> float:
    """
    Compares the color distribution of the raw lasso crop against the refined output.

    1.0  → identical color palettes (Gemini faithfully reproduced the source)
    ~0   → completely different colors (Gemini hallucinated or extended with wrong content)

    Algorithm:
      - Raw: extract non-white pixels (the actual part pixels, ignoring white bg)
      - Refined: extract non-transparent pixels
      - Build a 3D RGB histogram (16 bins/channel) for each set
      - Score = histogram intersection (sum of per-bin minimums)

    Works because characters have a distinctive color palette. If Gemini draws
    the wrong body part or invents new colors/textures the histograms diverge.
    Note: a part that is faithfully *extended* (same colors, just larger) will
    still score high here — use area_ratio alongside this to catch that case.
    """
    if not os.path.isfile(raw_path) or not os.path.isfile(refined_path):
        return 0.0

    raw_arr = np.array(Image.open(raw_path).convert("RGB"))
    ref_arr = np.array(Image.open(refined_path).convert("RGBA"))

    # Raw: ignore near-white background pixels
    not_white = ~(
        (raw_arr[:, :, 0] > 230)
        & (raw_arr[:, :, 1] > 230)
        & (raw_arr[:, :, 2] > 230)
    )
    raw_pixels = raw_arr[not_white].astype(np.float32) / 255.0  # (N, 3)

    # Refined: non-transparent pixels only
    non_transparent = ref_arr[:, :, 3] > 127
    ref_pixels = ref_arr[non_transparent, :3].astype(np.float32) / 255.0  # (M, 3)

    if len(raw_pixels) < 10 or len(ref_pixels) < 10:
        return 0.0

    edges = [np.linspace(0, 1, bins + 1)] * 3

    h_raw, _ = np.histogramdd(raw_pixels, bins=edges)
    h_ref, _ = np.histogramdd(ref_pixels, bins=edges)

    # Normalise so each sums to 1
    h_raw = h_raw / (h_raw.sum() + 1e-9)
    h_ref = h_ref / (h_ref.sum() + 1e-9)

    return float(np.minimum(h_raw, h_ref).sum())


# ── Per-part scoring ───────────────────────────────────────────────────────────


def score_part(png_path: str, polygon: list, raw_path: str | None = None) -> dict:
    """
    Score a single part PNG.
    raw_path: path to the raw lasso crop; if provided, color_fidelity is included.
    Returns a dict with individual metric scores and the composite mean.
    """
    img = Image.open(png_path).convert("RGBA")
    arr = np.array(img)

    poly_area = _polygon_area_px(polygon)

    s_cc    = score_connected_components(arr)
    s_holes = score_interior_holes(arr)
    s_area  = score_area_ratio(arr, poly_area)
    s_white = score_white_residue(arr)

    scores = {
        "connected_components": round(s_cc,    4),
        "interior_holes":       round(s_holes, 4),
        "area_ratio":           round(s_area,  4),
        "white_residue":        round(s_white, 4),
    }

    if raw_path is not None:
        s_fidelity = score_color_fidelity(raw_path, png_path)
        scores["color_fidelity"] = round(s_fidelity, 4)
        composite = (s_cc + s_holes + s_area + s_white + s_fidelity) / 5.0
    else:
        composite = (s_cc + s_holes + s_area + s_white) / 4.0

    scores["composite"] = round(composite, 4)
    return scores


# ── Character scoring ──────────────────────────────────────────────────────────


def score_character(
    character: str,
    lasso_dir: str,
    output_dir: str,
    use_raw: bool = False,
) -> dict | None:
    """
    Score all parts for a character.
    Returns a dict with per-part scores and an overall mean, or None if skipped.
    """
    lasso_json = os.path.join(lasso_dir, character, f"{character}_lasso_coords.json")
    if not os.path.isfile(lasso_json):
        candidates = [
            f for f in os.listdir(os.path.join(lasso_dir, character))
            if f.endswith("_lasso_coords.json")
        ] if os.path.isdir(os.path.join(lasso_dir, character)) else []
        if not candidates:
            print(f"[{character}] No lasso JSON found — skipping")
            return None
        lasso_json = os.path.join(lasso_dir, character, candidates[0])

    with open(lasso_json, encoding="utf-8") as f:
        lasso_data = json.load(f)

    parts = lasso_data.get("parts", [])
    if not parts:
        print(f"[{character}] No parts in JSON — skipping")
        return None

    part_subdir = "raw" if use_raw else "refined"
    parts_dir = os.path.join(output_dir, character, part_subdir)
    if not os.path.isdir(parts_dir):
        print(f"[{character}] Output dir not found: {parts_dir} — skipping")
        return None

    part_scores = {}
    all_composites = []

    for part_entry in parts:
        label   = part_entry["label"]
        polygon = part_entry["polygon"]
        png_path = os.path.join(parts_dir, f"{label}.png")

        if not os.path.isfile(png_path):
            print(f"  [{label}] PNG not found — skipping")
            continue

        # Always pass raw path for color_fidelity (only meaningful for refined outputs)
        raw_png  = os.path.join(output_dir, character, "raw", f"{label}.png")
        raw_path = raw_png if (not use_raw and os.path.isfile(raw_png)) else None

        scores = score_part(png_path, polygon, raw_path=raw_path)
        part_scores[label] = scores
        all_composites.append(scores["composite"])

        fidelity_str = (
            f"  fidelity={scores['color_fidelity']:.3f}"
            if "color_fidelity" in scores
            else ""
        )
        print(
            f"  [{label}]  "
            f"cc={scores['connected_components']:.3f}  "
            f"holes={scores['interior_holes']:.3f}  "
            f"area={scores['area_ratio']:.3f}  "
            f"white={scores['white_residue']:.3f}"
            f"{fidelity_str}  "
            f"→ {scores['composite']:.3f}"
        )

    if not all_composites:
        print(f"[{character}] No scored parts")
        return None

    overall = sum(all_composites) / len(all_composites)
    print(f"[{character}] overall={overall:.4f}  ({len(all_composites)} parts)\n")

    return {"parts": part_scores, "overall": overall}


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score lasso_batch.py outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "characters",
        nargs="*",
        metavar="CHARACTER",
        help="Characters to score. Omit to score all found in --output-dir.",
    )
    parser.add_argument(
        "--lasso-dir",
        default="resources/lasso",
        metavar="DIR",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        metavar="DIR",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Score raw crops instead of refined outputs",
    )
    args = parser.parse_args()

    repo_root  = os.path.dirname(os.path.abspath(__file__))
    lasso_dir  = os.path.join(repo_root, args.lasso_dir)
    output_dir = os.path.join(repo_root, args.output_dir)

    if not os.path.isdir(lasso_dir):
        sys.exit(f"Lasso directory not found: {lasso_dir}")

    if args.characters:
        characters = args.characters
    else:
        # Discover characters that have both a lasso JSON and output PNGs
        characters = []
        if os.path.isdir(lasso_dir):
            for d in sorted(os.listdir(lasso_dir)):
                char_out = os.path.join(output_dir, d, "raw" if args.raw else "refined")
                if os.path.isdir(char_out):
                    characters.append(d)

    if not characters:
        sys.exit(
            "No characters found with outputs. "
            "Run python lasso_batch.py first, then evaluate."
        )

    label = "raw" if args.raw else "refined"
    print(f"Evaluating {label} outputs for: {', '.join(characters)}\n")

    all_scores = {}
    for character in characters:
        result = score_character(
            character=character,
            lasso_dir=lasso_dir,
            output_dir=output_dir,
            use_raw=args.raw,
        )
        if result is not None:
            all_scores[character] = result["overall"]

    if not all_scores:
        sys.exit("No characters could be scored.")

    grand_mean = sum(all_scores.values()) / len(all_scores)

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for char, score in sorted(all_scores.items()):
        print(f"  {char:<20} {score:.4f}")
    print("-" * 50)
    print(f"  {'OVERALL':<20} {grand_mean:.4f}")
    print("=" * 50)
    print(f"\noverall_score: {grand_mean:.6f}")


if __name__ == "__main__":
    main()
