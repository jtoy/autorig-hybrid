import os
import json
import io
import base64
import shutil
import subprocess
from PIL import Image
from google.genai import types


PART_NAME_MAP = {
    'head': 'head',
    'torso': 'torso',
    'left_upperarm': 'leftUpperArm',
    'right_upperarm': 'rightUpperArm',
    'left_forearm': 'leftForearm',
    'right_forearm': 'rightForearm',
    'left_hand': 'leftHand',
    'right_hand': 'rightHand',
    'left_thigh': 'leftThigh',
    'right_thigh': 'rightThigh',
    'left_calf': 'leftLeg',
    'right_calf': 'rightLeg',
    'left_foot': 'leftFoot',
    'right_foot': 'rightFoot',
}

TANK_EXAMPLE = """{
  "dimensionValues": {
    "head": {"width": 337, "height": 437},
    "torso": {"width": 278, "height": 383},
    "leftUpperArm": {"width": 88, "height": 203},
    "rightUpperArm": {"width": 83, "height": 203},
    "leftForearm": {"width": 58, "height": 135},
    "rightForearm": {"width": 59, "height": 137},
    "leftHand": {"width": 45, "height": 50},
    "rightHand": {"width": 45, "height": 50},
    "leftThigh": {"width": 111, "height": 198},
    "rightThigh": {"width": 80, "height": 166},
    "leftLeg": {"width": 184, "height": 137},
    "rightLeg": {"width": 184, "height": 137},
    "leftFoot": {"width": 60, "height": 40},
    "rightFoot": {"width": 60, "height": 40}
  },
  "pivotPoints": {
    "torso_head": {"x": 40, "y": -343},
    "torso_leftUpperArm": {"x": -56, "y": -309},
    "torso_rightUpperArm": {"x": 76, "y": -322},
    "torso_leftThigh": {"x": -22, "y": -67},
    "torso_rightThigh": {"x": 85, "y": -58},
    "leftUpperArm_leftForearm": {"x": 4, "y": 60},
    "rightUpperArm_rightForearm": {"x": -9, "y": 49},
    "leftForearm_leftHand": {"x": 0, "y": 50},
    "rightForearm_rightHand": {"x": 0, "y": 50},
    "leftThigh_leftLeg": {"x": 0, "y": 40},
    "rightThigh_rightLeg": {"x": 9, "y": 36},
    "leftLeg_leftFoot": {"x": 0, "y": 30},
    "rightLeg_rightFoot": {"x": 0, "y": 30}
  },
  "jointOffset": {
    "torso_head": {"x": 19, "y": 33},
    "torso_leftUpperArm": {"x": 6, "y": 40},
    "torso_rightUpperArm": {"x": 0, "y": 26},
    "torso_leftThigh": {"x": 0, "y": 22},
    "torso_rightThigh": {"x": 0, "y": 22},
    "leftUpperArm_leftForearm": {"x": 0, "y": 19},
    "rightUpperArm_rightForearm": {"x": 0, "y": 19},
    "leftForearm_leftHand": {"x": 0, "y": 15},
    "rightForearm_rightHand": {"x": 0, "y": 15},
    "leftThigh_leftLeg": {"x": 0, "y": 13},
    "rightThigh_rightLeg": {"x": 0, "y": 13},
    "leftLeg_leftFoot": {"x": 0, "y": 10},
    "rightLeg_rightFoot": {"x": 0, "y": 10}
  },
  "zIndexValues": {
    "head": 9, "torso": 8,
    "leftUpperArm": 10, "leftForearm": 11, "leftHand": 12,
    "rightUpperArm": 1, "rightForearm": 2, "rightHand": 3,
    "leftThigh": 6, "leftLeg": 7, "leftFoot": 13,
    "rightThigh": 4, "rightLeg": 5, "rightFoot": 14
  }
}"""


def _parse_json(text):
    if not text:
        raise ValueError("Empty response from Gemini")
    lines = text.strip().splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            text = "\n".join(lines[i + 1:])
            text = text.split("```")[0]
            break
    return json.loads(text)


def trim_transparent(img, threshold=20):
    """Crop RGBA image to its opaque bounding box (alpha >= threshold)."""
    img = img.convert("RGBA")
    alpha = img.split()[3]
    alpha = alpha.point(lambda a: 0 if a < threshold else a)
    bbox = alpha.getbbox()
    if bbox is None:
        return img
    return img.crop(bbox)


def generate_rig_params(client, original_image_path, model: str = "gemini-2.5-flash"):
    """
    Show Gemini ONLY the original character image + Tank reference.
    Have it produce all rig params by analyzing the character's proportions visually.
    """
    original = Image.open(original_image_path)
    ow, oh = original.size

    prompt = f"""You are a 2D character rigging expert. Look at this character image ({ow}x{oh}px) and produce rig parameters.

COORDINATE SYSTEM:
- Torso anchor is at BOTTOM-CENTER (0.5, 1.0). Root transform = torso bottom-center.
- pivotPoints are offsets FROM the root. Negative y = above root.
- For torso_head: if neck is 40px from top of a 383px torso → pivot_y = -(383-40) = -343
- For limb-to-limb (elbow/knee/wrist/ankle): pivot_y = parent_height - distance_from_top_to_joint
- jointOffset pushes child images to OVERLAP the parent at the joint. Must be generous (15-25% of child height) so parts connect seamlessly with NO visible gaps.
- Target torso height: ~300px. Scale all other parts proportionally from the character image.
- Upper arms/thighs typically 50-55% of torso height.
- Forearms/calves typically 35-40% of torso height.
- Hands/feet typically 15-20% of torso height.
- Must fit 1000x1000 canvas (root at ~500,600).

CRITICAL — ARM AND LEG PLACEMENT:
- Shoulder pivot x MUST be at the OUTER EDGES of the torso so arms hang at the SIDES, NOT overlapping the chest.
- torso_leftUpperArm x should be approximately -(torso_width * 0.45 to 0.55) — at or beyond the left edge.
- torso_rightUpperArm x should be approximately +(torso_width * 0.45 to 0.55) — at or beyond the right edge.
- Hip pivot x should place thighs under the torso edges, not centered.
- torso_leftThigh x should be approximately -(torso_width * 0.20 to 0.30).
- torso_rightThigh x should be approximately +(torso_width * 0.20 to 0.30).
- The arms MUST be visible at the left and right sides of the body when assembled.

REFERENCE — working rig for a cartoon character "Tank" (torso 278x383):
{TANK_EXAMPLE}

Note in the Tank reference:
- torso_leftUpperArm x=-56 with torso_width=278 → x/half_width = 56/139 = 0.40 (barely minimum)
- torso_rightUpperArm x=76 → x/half_width = 76/139 = 0.55 (good)
- You should aim for 0.45-0.55 of half-torso-width for shoulder x offsets.

Look at the character image and determine:
1. The CORRECT proportional dimensions of each body part (head, torso, arms, forearms, hands, thighs, calves, feet)
2. Where joints connect (neck, shoulders, hips, elbows, wrists, knees, ankles)
3. Generous joint offsets so assembled parts overlap seamlessly
4. Shoulder pivots far enough out so arms are AT THE SIDES of the torso

Return JSON with exactly these 4 keys: "dimensionValues", "pivotPoints", "jointOffset", "zIndexValues".
Include all 14 parts: head, torso, leftUpperArm, rightUpperArm, leftForearm, rightForearm, leftHand, rightHand, leftThigh, rightThigh, leftLeg, rightLeg, leftFoot, rightFoot.
Include all 13 pivot/offset joints: torso_head, torso_leftUpperArm, torso_rightUpperArm, torso_leftThigh, torso_rightThigh, leftUpperArm_leftForearm, rightUpperArm_rightForearm, leftForearm_leftHand, rightForearm_rightHand, leftThigh_leftLeg, rightThigh_rightLeg, leftLeg_leftFoot, rightLeg_rightFoot."""

    print(f"[rig] Generating rig params with {model} for {ow}x{oh}px image...")
    response = client.models.generate_content(
        model=model,
        contents=[original, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        )
    )
    result = _parse_json(response.text)
    print(f"[rig] Got params: {len(result.get('dimensionValues', {}))} dimensions, "
          f"{len(result.get('pivotPoints', {}))} pivots")
    return result


def rig(client, original_image_path, parts_dir, output_path, model: str = "gemini-2.5-flash"):
    """
    Generate a distark rig JSON from die-cut body parts.

    1. Trim transparent padding from part images
    2. Encode trimmed images as base64 data URLs
    3. Use Gemini on the ORIGINAL character image to determine all rig params
    4. Assemble final rig JSON
    """
    # Step 1: Load, trim, and encode part images as base64
    image_paths = {}
    for file_name, rig_name in PART_NAME_MAP.items():
        path = os.path.join(parts_dir, f"{file_name}.png")
        if os.path.exists(path):
            img = Image.open(path).convert("RGBA")
            trimmed = trim_transparent(img)
            buf = io.BytesIO()
            trimmed.save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            image_paths[rig_name] = f"data:image/png;base64,{b64}"

    print(f"Found {len(image_paths)} parts: {list(image_paths.keys())}")

    # Step 2: Gemini analyzes the ORIGINAL image for all rig parameters
    print("Asking Gemini to analyze original image for rig parameters...")
    params = generate_rig_params(client, original_image_path, model=model)

    dims = params['dimensionValues']
    pivot_points = params['pivotPoints']
    joint_offset = params['jointOffset']
    z_index_values = params['zIndexValues']

    # Post-process: ensure shoulder pivots are far enough out that arms are
    # visible at the sides. The behind-arm (low z-index) must extend past
    # the torso edge to be visible.
    torso_w = dims['torso']['width']
    half_w = torso_w / 2.0
    min_shoulder_ratio = 0.85  # arms near outer edges of torso

    for key, sign in [('torso_leftUpperArm', -1), ('torso_rightUpperArm', 1)]:
        if key in pivot_points:
            current_x = pivot_points[key]['x']
            min_x = sign * half_w * min_shoulder_ratio
            if abs(current_x) < abs(min_x):
                print(f"Adjusting {key} x from {current_x} to {round(min_x)} (was too centered)")
                pivot_points[key]['x'] = round(min_x)

    # Post-process: ensure joint offsets are generous enough for seamless assembly
    # Minimum y-offset as fraction of child height
    min_joint_y_ratios = {
        'torso_head': 0.10,
        'torso_leftUpperArm': 0.25,
        'torso_rightUpperArm': 0.25,
        'torso_leftThigh': 0.20,
        'torso_rightThigh': 0.20,
        'leftUpperArm_leftForearm': 0.20,
        'rightUpperArm_rightForearm': 0.20,
        'leftForearm_leftHand': 0.15,
        'rightForearm_rightHand': 0.15,
        'leftThigh_leftLeg': 0.15,
        'rightThigh_rightLeg': 0.15,
        'leftLeg_leftFoot': 0.10,
        'rightLeg_rightFoot': 0.10,
    }
    # Map joint keys to the child part for height lookup
    joint_child_map = {
        'torso_head': 'head',
        'torso_leftUpperArm': 'leftUpperArm',
        'torso_rightUpperArm': 'rightUpperArm',
        'torso_leftThigh': 'leftThigh',
        'torso_rightThigh': 'rightThigh',
        'leftUpperArm_leftForearm': 'leftForearm',
        'rightUpperArm_rightForearm': 'rightForearm',
        'leftForearm_leftHand': 'leftHand',
        'rightForearm_rightHand': 'rightHand',
        'leftThigh_leftLeg': 'leftLeg',
        'rightThigh_rightLeg': 'rightLeg',
        'leftLeg_leftFoot': 'leftFoot',
        'rightLeg_rightFoot': 'rightFoot',
    }
    for jkey, min_ratio in min_joint_y_ratios.items():
        if jkey in joint_offset and jkey in joint_child_map:
            child = joint_child_map[jkey]
            if child in dims:
                child_h = dims[child]['height']
                min_y = int(child_h * min_ratio)
                if joint_offset[jkey]['y'] < min_y:
                    print(f"Adjusting {jkey} jointOffset y from {joint_offset[jkey]['y']} to {min_y}")
                    joint_offset[jkey]['y'] = min_y

    print(f"Dimensions: {json.dumps(dims, indent=2)}")

    # Step 3: Fixed rotation/selfRotation values
    rotation_values = {
        'head': 0.0, 'torso': 0.0,
        'leftUpperArm': -3.14159265358979, 'rightUpperArm': 3.14159265358979,
        'leftForearm': 0.0, 'rightForearm': 0.0,
        'leftHand': 0.0, 'rightHand': 0.0,
        'leftThigh': -3.14159265358979, 'rightThigh': 3.14159265358979,
        'leftLeg': 0.0, 'rightLeg': 0.0,
        'leftFoot': 0.0, 'rightFoot': 0.0,
    }
    self_rotation_values = {
        'head': 0.0, 'torso': 0.0,
        'leftUpperArm': -3.14, 'rightUpperArm': -3.14,
        'leftForearm': -3.14, 'rightForearm': -3.14,
        'leftHand': -3.14, 'rightHand': -3.14,
        'leftThigh': -3.14, 'rightThigh': -3.14,
        'leftLeg': -3.14, 'rightLeg': -3.14,
        'leftFoot': -3.14, 'rightFoot': -3.14,
    }

    # Step 4: Hide parts not produced by turtlediecutter
    visibility = {}
    for name in ['mouth']:
        if name not in image_paths:
            visibility[name] = False

    # Step 5: Assemble rig JSON
    rig_data = {
        'kind': 'character',
        'imageScale': 1,
        'imagePaths': image_paths,
        'dimensionValues': dims,
        'rotationValues': rotation_values,
        'selfRotationValues': self_rotation_values,
        'pivotPoints': pivot_points,
        'jointOffset': joint_offset,
        'zIndexValues': z_index_values,
        'visibility': visibility,
    }

    with open(output_path, 'w') as f:
        json.dump(rig_data, f, indent=2)

    print(f"Rig saved to {output_path}")

    # Validate, render, and visually test with distark-check
    validate_rig(output_path, original_image_path)

    return output_path


def _find_distark_check():
    """Find distark-check binary: global PATH, then local node_modules."""
    path = shutil.which("distark-check")
    if path:
        return path
    local = os.path.join("node_modules", ".bin", "distark-check")
    if os.path.isfile(local) and os.access(local, os.X_OK):
        return local
    return None


def validate_rig(rig_path, original_image_path=None):
    """Run distark-check verify, render, and test on the rig JSON."""
    cmd = _find_distark_check()
    if not cmd:
        print(
            "WARNING: distark-check not found. Install it to validate rigs:\n"
            "  npm install distark-render\n"
            "  # or globally: npm install -g distark-render"
        )
        return None

    # 1. verify — math-only sanity checks
    try:
        result = subprocess.run(
            [cmd, "verify", rig_path],
            capture_output=True,
            text=True,
        )
        report = json.loads(result.stdout) if result.stdout else {}

        failed = report.get("failed", [])
        warnings = report.get("warnings", [])
        passed = report.get("passed", [])

        if failed:
            print(f"distark-check verify FAILED ({len(failed)} issues):")
            for item in failed:
                print(f"  FAIL: [{item['check']}] {item.get('detail', '')}")
        if warnings:
            print(f"distark-check verify warnings ({len(warnings)}):")
            for w in warnings:
                print(f"  WARN: [{w['check']}] {w.get('detail', '')}")
        if passed and not failed:
            print(f"distark-check verify passed: {', '.join(passed)}")
    except Exception as e:
        print(f"WARNING: distark-check verify failed to run: {e}")
        report = None

    # 2. render — generate a PNG so the user can review
    render_path = rig_path.rsplit('.', 1)[0] + '.png'
    try:
        result = subprocess.run(
            [cmd, "render", rig_path, "-o", render_path],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"Rendered rig to {render_path}")
        else:
            print(f"WARNING: distark-check render failed: {result.stderr.strip()}")
    except Exception as e:
        print(f"WARNING: distark-check render failed to run: {e}")

    # 3. test — visual assertion comparing rendered rig to original
    if original_image_path and os.path.isfile(render_path):
        try:
            prompt = (
                "Does this image show a properly assembled character with "
                "head, torso, arms, and legs all connected and visible? "
                "Are the body parts arranged in a natural pose with no "
                "overlapping limbs hiding each other?"
            )
            result = subprocess.run(
                [cmd, "test", render_path, "--prompt", prompt],
                capture_output=True,
                text=True,
            )
            test_report = json.loads(result.stdout) if result.stdout else {}
            passed = test_report.get("pass", False)
            reason = test_report.get("reason", "")
            status = "PASSED" if passed else "FAILED"
            print(f"distark-check test {status}: {reason}")
        except Exception as e:
            print(f"WARNING: distark-check test failed to run: {e}")

    return report
