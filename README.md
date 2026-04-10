# Auto-Rig

AI-powered character auto-rigging pipeline. It takes a 2D character image, cuts it into rig parts with Gemini-assisted processing, and assembles a [distark-render](https://libraries.io/npm/distark-render)-compatible rig JSON that can be previewed in the included FastAPI UI.

## Problem Statement

This project is currently focused on two quality problems:

1. **Cut the pieces accurately** - The pipeline should separate the character into the correct body-part pieces with clean boundaries, and it should also fill missing or occluded regions so each exported part is visually complete and usable in the final rig.
2. **Fix scaling** - The generated parts and assembled rig should preserve the same apparent scale as the original uploaded image, instead of producing pieces that are too large, too small, or inconsistent with the source character proportions.

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

- **Python 3.10+**
- **Node 18.x** if you want optional `distark-render` rig validation
- **Gemini API key** - get one at https://ai.google.dev/gemini-api/docs/api-key

## Install

### 1. Clone and enter the repo

```bash
git clone <your-repo-url>
cd autorig-private
```

### 2. Use the Python environment

This workspace convention uses the `conjurors` conda environment:

```bash
conda activate conjurors
```

If you are not using that shared environment, create a local virtual environment instead:

```bash
python -m venv .venv
```

Activate it:

```bash
# PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Install Python dependencies

The UI imports a few packages beyond the current `requirements.txt`, so install both the listed dependencies and the UI/runtime extras:

```bash
pip install -r requirements.txt fastapi uvicorn python-multipart python-dotenv
```

### 4. Configure the Gemini API key

You can either export the variable in your shell:

```bash
# PowerShell
$env:GEMINI_API_KEY="your_key_here"

# Linux / macOS
export GEMINI_API_KEY="your_key_here"
```

Or create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_key_here
```

### 5. Optional: install rig validation tooling

If you want `distark-check verify`, `distark-check render`, and `distark-check test`, use Node 18 and install `distark-render`:

```bash
nvm use 18
yarn add distark-render
```

## Run

After installing dependencies and setting `GEMINI_API_KEY`, start the FastAPI UI:

```bash
python ui/server.py
```

Or, equivalently:

```bash
python -m uvicorn ui.server:app --host 0.0.0.0 --port 8888
```

Then open [http://localhost:8888](http://localhost:8888).

### Quick Start

1. Start the server with `python ui/server.py`.
2. Open `http://localhost:8888` in your browser.
3. Upload a sample image from `resources/`, for example `resources/hippo.png`.
4. Click **Run Pipeline** to execute diecut, bounding boxes, background cleanup, and rig assembly.
5. Review the extracted parts on the left and the assembled rig preview on the right.
6. Edit pieces or pivots if needed, then regenerate or download the resulting `rig.json`.

## Usage

### Option A: Web UI (Recommended)

The web UI provides an interactive interface for running the pipeline, editing individual parts, adjusting rig parameters, and previewing the assembled character with animation.

```bash
python -m uvicorn ui.server:app --host 0.0.0.0 --port 8888
```

Then open http://localhost:8888 in your browser.

By default there is **no authentication**. To enable HTTP Basic Auth, set both environment variables:
```bash
# PowerShell
$env:AUTH_USER="your_username"
$env:AUTH_PASS="your_password"

# Linux / macOS
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

- **Piece cutting and completion** - extracted pieces can still miss edges, merge incorrectly, or fail to reconstruct hidden portions cleanly
- **Scaling fidelity** - the assembled rig can drift away from the original image scale and proportions
- **Session cleanup** - the web UI creates temp directories that are not automatically cleaned up
- **Base64 rig size** - rig JSON files can be 10-15 MB due to embedded base64 images

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
