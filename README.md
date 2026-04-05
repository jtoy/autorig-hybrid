# Auto-Rig

AI-powered character auto-rigging pipeline. Takes a 2D character image, separates it into body parts using Gemini, and assembles a [distark-render](https://libraries.io/npm/distark-render)-compatible rig JSON.

## What It Does

1. **Diecut** — Gemini generates a composite image with the character separated into 14 body parts
2. **Bboxes** — Gemini detects bounding boxes for each part and crops them into individual PNGs
3. **Background Removal** — Flood-fill algorithm removes backgrounds from each part
4. **Rig Assembly** — Gemini analyzes proportions and assembles the final rig JSON with pivot points, dimensions, z-ordering, and rotations

The output `rig.json` is compatible with the `distark-render` npm package for rendering and animation.

## Target Rig Format

A reference rig looks like this: https://cln.sh/LjsQBqyDNcnZ5gPbSZGm

The rig JSON contains:
- `imagePaths` — base64-encoded PNGs for 10 body parts (head, torso, leftUpperArm, rightUpperArm, leftForearm, rightForearm, leftThigh, rightThigh, leftLeg, rightLeg)
- `dimensionValues` — pixel dimensions for each part
- `pivotPoints` — joint offsets relative to parent (torso is root)
- `jointOffset` — overlap amounts for seamless assembly
- `rotationValues` / `selfRotationValues` — animation keyframes
- `zIndexValues` — draw order for correct layering
- `visibility` — which parts are visible

## Requirements

- **Python 3.10+** (tested with 3.13)
- **Node 18+** (for optional `distark-render` validation)
- **Gemini API key** — get one free at https://ai.google.dev/gemini-api/docs/api-key

## Setup

### 1. Clone and enter the repo

```bash
git clone https://github.com/jtoy/autorig.git
cd autorig
```

### 2. Create a Python environment

Using conda:
```bash
conda create -n autorig python=3.13
conda activate autorig
```

Or using venv:
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Gemini API key

```bash
export GEMINI_API_KEY="your_key_here"
```

Or create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```

### 5. (Optional) Install distark-render for rig validation

```bash
npm install distark-render
```

This enables `distark-check verify`, `distark-check render`, and `distark-check test` commands used during rig generation to validate output.

## Quick Start

After completing the setup above, here's the fastest way to see the pipeline in action:

```bash
# 1. Install dependencies (if you haven't already)
pip install -r requirements.txt

# 2. Start the web UI
python -m uvicorn ui.server:app --host 0.0.0.0 --port 8888
```

3. Open http://localhost:8888 in your browser
4. Click **"Choose File"** and upload one of the test images from `resources/` (e.g. `resources/hippo.png`)
5. Click **"Run Pipeline"** — this runs all 4 steps (diecut → bboxes → background removal → rig). It takes 1-3 minutes depending on the Gemini model and number of refinement rounds.
6. When it finishes, you'll see:
   - The **cropped body parts** in the parts panel on the left
   - The **assembled rig preview** on the right, showing the character reconstructed from its parts
   - You can click parts to edit them, adjust pivot points, toggle animation, and compare against the original image

## Usage

### Option A: Web UI (Recommended)

The web UI provides an interactive interface for running the pipeline, editing individual parts, adjusting rig parameters, and previewing the assembled character with animation.

```bash
python -m uvicorn ui.server:app --host 0.0.0.0 --port 8888
```

Then open http://localhost:8888 in your browser.

By default there is **no authentication**. To enable HTTP Basic Auth, set both environment variables:
```bash
export AUTH_USER="your_username"
export AUTH_PASS="your_password"
```

If either is unset or empty, the server runs without authentication.

#### Web UI Workflow

1. **Upload** — Click "Choose File" and upload a character PNG image (test images are in the `resources/` folder — try `hippo.png`, `tobyturtle.png`, `papa.png`, or `grandpa.png`)
2. **Run Pipeline** — Click "Run Pipeline" to execute all 4 steps automatically, or run each step individually:
   - **Diecut** — generates the composite diecut image
   - **Bboxes** — detects and crops the 14 body parts
   - **Remove BG** — removes backgrounds from all parts
   - **Rig** — assembles the final rig JSON
3. **Review Parts** — Inspect each cropped body part in the parts panel. You can:
   - Click a part to open it in the editor
   - Draw, erase, or fill to fix part images
   - Resize parts
   - Upload replacement PNGs
4. **Adjust Rig** — In the rig preview panel you can:
   - Toggle between View, Pivots, and Parts edit modes
   - Adjust pivot points, dimensions, rotations, and z-index values
   - Overlay a reference rig for comparison
   - Compare the assembled rig against the original image
5. **Regenerate Rig** — After editing parts, click "Regenerate Rig" to re-run rig assembly with updated parts (without re-running diecut/bboxes)
6. **Download** — Save the final `rig.json`

#### Web UI Configuration

The UI has model selector dropdowns to configure which Gemini models are used for each step:
- **Diecut model** — for generating the composite diecut image
- **Vision model** — for the diecut judge and bboxes detection
- **Rig model** — for rig assembly analysis

The diecut step also has a configurable **rounds** slider (1-10) that controls how many refinement iterations the judge runs.

### Option B: CLI Agent

Run the full pipeline from the command line using the LangChain agent:

```bash
python agent.py --image resources/hippo.png --diecut-output diecut.png --pieces-dir parts
```

**Arguments:**
- `--image` — path to the input character image (default: `resources/hippo.png`)
- `--diecut-output` — path for the composite diecut image (default: `diecut.png`)
- `--pieces-dir` — folder for cropped part PNGs (default: `parts`)

**Output:**
- `diecut.png` — composite image with all 14 parts separated
- `parts/` — 14 individual body part PNGs (head, torso, left_upperarm, etc.)
- `rig.json` — distark-compatible rig file

### Option C: Individual Steps

You can also run each processing step independently:

```bash
# Just generate rig from existing parts
python rig.py --image original.png --parts-dir parts --output rig.json

# Generate rig without Gemini (proportional fallback)
python rig.py --image original.png --parts-dir parts --no-gemini
```

## Project Structure

```
├── agent.py                  # LangChain agentic orchestrator (CLI entry point)
├── requirements.txt          # Python dependencies
├── processing/               # Core pipeline modules
│   ├── __init__.py           #   Public exports
│   ├── diecut.py             #   AI diecut generation with judge loop
│   ├── bboxes.py             #   Body part detection & cropping
│   ├── rig.py                #   Rig JSON assembly & validation
│   ├── simple_background.py  #   Fast flood-fill background removal
│   ├── genai_background.py   #   GenAI-based background removal
│   ├── background.py         #   Background color changer
│   └── alpha.py              #   Alpha channel extraction (two-pass)
├── ui/                       # Web application
│   ├── server.py             #   FastAPI backend
│   └── static/
│       ├── index.html        #   UI layout
│       ├── app.js            #   Client-side logic
│       └── style.css         #   Styling
├── resources/                # Example input images & reference diecuts
│   ├── hippo.png
│   ├── tobyturtle.png
│   ├── tobyturtle-diecut2.png
│   └── ...
└── deploy.sh                 # Remote deployment script
```

## Pipeline Details

### Body Parts (14 total)

The pipeline detects and separates these parts:
- **Head** (1)
- **Torso** (1)
- **Arms** (6): left/right × upperarm, forearm, hand
- **Legs** (6): left/right × thigh, calf, foot

Parts use `snake_case` filenames internally (e.g. `left_upperarm.png`). The rig JSON uses `camelCase` keys (e.g. `leftUpperArm`).

Note: The final rig maps 14 detected parts to 10 rig slots (hands merge with forearms, feet merge with calves).

### Diecut Judge

The diecut step runs a multi-round refinement loop. Each round:
1. Gemini generates/refines the composite diecut image
2. A judge model evaluates 5 criteria:
   - **part_count** — exactly 14 separate pieces
   - **segmentation** — arms split into 3 pieces, legs split into 3 pieces
   - **style_fidelity** — original style/colors preserved
   - **detail_preservation** — hands/feet unmodified
   - **layout** — parts in a row, no overlaps
3. Failed criteria are fed back as targeted refinement instructions
4. The best scoring attempt across all rounds is kept

### Rig Assembly

The rig step:
1. Trims transparency from each part PNG
2. Sends parts + original image to Gemini for proportion analysis
3. Generates dimension values, pivot points, joint overlaps, and z-ordering
4. Validates shoulder placement (arms at sides, not centered)
5. Validates joint overlaps (minimum 10-25% depending on part)
6. Runs `distark-check verify` if installed

## Known Issues / Areas for Improvement

- **Part sizing and positioning** — generated parts are sometimes the wrong size or placed incorrectly relative to the original character proportions
- **Session cleanup** — the web UI creates temp directories that are not automatically cleaned up
- **Base64 rig size** — rig JSON files can be 10-15 MB due to embedded base64 images

## Example Source Images

Test images are in `resources/`. Additional test characters:
- https://cln.sh/cJfLnxtnmgmydWVlGlhV

## Deployment

The `deploy.sh` script syncs files to a remote server via rsync and restarts the uvicorn process:

```bash
./deploy.sh
```

Requires SSH access configured for the remote host. Edit the `REMOTE` and `REMOTE_DIR` variables in `deploy.sh` to match your server.

## Tech Stack

- **Python**: Google GenAI SDK, LangChain, Pillow, NumPy
- **Web**: FastAPI + uvicorn, vanilla JS
- **AI**: Google Gemini (image generation, vision, analysis)
- **Rendering**: [distark-render](https://libraries.io/npm/distark-render) (npm)
