import argparse
import json
import os
from typing import Any, Type
from pydantic import BaseModel, Field

from google import genai

from processing import diecut
from processing import bboxes
from processing import change_background, change_alpha
from processing import rig
from processing.simple_background import remove_background_simple
from processing.genai_background import remove_background_genai

client = genai.Client()
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MISSING = object()

class DiecutInput(BaseModel):
    image_path: str = Field(description="Input image path (e.g., 'tobyturtle.png')")
    output_path: str = Field(description="Output path for the diecut image (e.g., 'diecut.png')")

class BboxesInput(BaseModel):
    image_path: str = Field(description="Input image path")
    output_folder: str = Field(description="Folder where cropped parts will be saved")

class RemoveBackgroundInput(BaseModel):
    pieces_dir: str = Field(description="Folder with PNG parts output from bboxes")
    use_genai: bool = Field(default=False, description="Use GenAI background removal (slower)")
    tolerance: int = Field(default=30, description="Color tolerance for local removal (0-255)")

class RigInput(BaseModel):
    original_image_path: str = Field(description="Path to the original character image")
    pieces_dir: str = Field(description="Folder with PNG body parts (output from bboxes + background removal)")
    output_path: str = Field(default="rig.json", description="Output path for the rig JSON file")


def removeBackground(pieces_dir: str, use_genai: bool = True, tolerance: int = 30):
    if not os.path.isdir(pieces_dir):
        raise FileNotFoundError(f"Parts folder not found: {pieces_dir}")

    for filename in os.listdir(pieces_dir):
        if not filename.lower().endswith(".png"):
            continue

        part_path = os.path.join(pieces_dir, filename)
        if use_genai:
            remove_background_genai(client, part_path, tolerance=tolerance)
        else:
            remove_background_simple(part_path, tolerance=tolerance)


def build_agent_tools():
    from langchain.tools import BaseTool

    class DiecutTool(BaseTool):
        name: str = "diecut"
        description: str = (
            "Create a single composite diecut image that preserves the original pose and proportions, "
            "with the character cleanly separated into these 14 parts in the same image: "
            "head, torso, right_upperarm, left_upperarm, right_forearm, left_forearm, right_hand, left_hand, "
            "right_thigh, left_thigh, right_calf, left_calf, right_foot, left_foot. "
            "Input: image_path. Output: diecut composite saved to output_path."
        )
        args_schema: Type[BaseModel] = DiecutInput

        def _run(self, image_path: str, output_path: str):
            try:
                diecut(client, image_path, output_path)
                return f"Diecut image saved successfully to {output_path}"
            except Exception as e:
                return f"Error running diecut: {str(e)}"

    class BboxesTool(BaseTool):
        name: str = "bboxes"
        description: str = (
            "Detects exactly 14 body parts in the given image and writes PNG crops into output_folder. "
            "Output filenames are the labels in English: head, torso, right_upperarm, left_upperarm, "
            "right_forearm, left_forearm, right_hand, left_hand, right_thigh, left_thigh, right_calf, "
            "left_calf, right_foot, left_foot (e.g., head.png, right_hand.png)."
        )
        args_schema: Type[BaseModel] = BboxesInput

        def _run(self, image_path: str, output_folder: str):
            try:
                bboxes(client, image_path, output_folder)
                return f"Part crops saved successfully to folder {output_folder}"
            except Exception as e:
                return f"Error running bboxes: {str(e)}"

    class RemoveBackgroundTool(BaseTool):
        name: str = "remove_background"
        description: str = (
            "Removes background for every PNG part inside pieces_dir. By default it uses a fast local "
            "color-based removal and overwrites each PNG in-place. If use_genai=true, it uses slower "
            "GenAI background passes."
        )
        args_schema: Type[BaseModel] = RemoveBackgroundInput

        def _run(self, pieces_dir: str, use_genai: bool = False, tolerance: int = 30):
            try:
                removeBackground(pieces_dir, use_genai=use_genai, tolerance=tolerance)
                return f"Background removed for all parts in {pieces_dir}"
            except Exception as e:
                return f"Error running removeBackground: {str(e)}"

    class RigTool(BaseTool):
        name: str = "rig"
        description: str = (
            "Generate a distark-compatible rig JSON from die-cut body parts. "
            "Uses Gemini to detect joint positions on the torso and limb images, "
            "then assembles a complete rig with pivot points, dimensions, rotations, and z-ordering. "
            "Automatically validates the output with distark-check verify if installed. "
            "Input: original_image_path, pieces_dir. Output: rig JSON saved to output_path."
        )
        args_schema: Type[BaseModel] = RigInput

        def _run(self, original_image_path: str, pieces_dir: str, output_path: str = "rig.json"):
            try:
                rig(client, original_image_path, pieces_dir, output_path)
                return f"Rig JSON saved successfully to {output_path}"
            except Exception as e:
                return f"Error running rig: {str(e)}"

    return [DiecutTool(), BboxesTool(), RemoveBackgroundTool(), RigTool()]

def agenticDiecut(image_path: str, diecut_output: str, pieces_dir: str):
    from langchain.agents import create_agent
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = "gemini-3-pro-preview"
    temperature = 0
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        convert_system_message_to_human=True
    )

    tools = build_agent_tools()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are an animator artist specializing in character die-cutting and rigging. "
            "You must execute the following steps SEQUENTIALLY. Wait for each tool to finish before calling the next: "
            "1. Call 'diecut' to create the composite image. "
            "2. AFTER 'diecut' succeeds, call 'bboxes' to extract the pieces into a folder. "
            "3. AFTER 'bboxes' succeeds, call 'remove_background' on that folder to clean the pieces. "
            "4. AFTER 'remove_background' succeeds, call 'rig' to generate a rig JSON from the cleaned parts. "
            "DO NOT call these tools in parallel in the same turn, as each depends on the previous one's output."
        ),
        debug=True
    )

    task = (
        f"Take the image '{image_path}', first apply diecut saving it as "
        f"'{diecut_output}', then extract the pieces from that result into the "
        f"folder '{pieces_dir}', then remove the background from all pieces in "
        f"that folder, then generate a rig JSON from the cleaned parts using "
        f"the original image '{image_path}' and pieces folder '{pieces_dir}'."
    )

    print(f"Running task: {task}")

    try:
        for update in agent.stream(
            {"messages": [{"role": "user", "content": task}]},
            stream_mode="updates",
        ):
            print(update)
    except Exception as e:
        print(f"Error during agent execution: {str(e)}")


def run_pipeline(
    image_path: str,
    work_dir: str,
    diecut_rounds: int = 5,
    diecut_model: str | None = None,
    vision_model: str | None = None,
    rig_model: str | None = None,
    use_genai_background: bool = False,
    tolerance: int = 30,
    fail_on_review: bool = False,
):
    """Run the full auto-rig pipeline for a single input image."""
    os.makedirs(work_dir, exist_ok=True)

    diecut_output = os.path.join(work_dir, "diecut.png")
    pieces_dir = os.path.join(work_dir, "parts")
    rig_output = os.path.join(work_dir, "rig.json")

    diecut_kwargs = {}
    if diecut_model:
        diecut_kwargs["model"] = diecut_model
    if vision_model:
        diecut_kwargs["judge_model"] = vision_model

    bbox_kwargs = {}
    if vision_model:
        bbox_kwargs["model"] = vision_model

    rig_kwargs = {}
    if rig_model:
        rig_kwargs["model"] = rig_model

    print(f"[pipeline] Processing {image_path}")
    diecut(client, image_path, diecut_output, fail_on_review, diecut_rounds, **diecut_kwargs)
    bboxes(client, diecut_output, pieces_dir, **bbox_kwargs)
    removeBackground(pieces_dir, use_genai=use_genai_background, tolerance=tolerance)
    rig(client, image_path, pieces_dir, rig_output, **rig_kwargs)

    return {
        "diecut_output": diecut_output,
        "pieces_dir": pieces_dir,
        "rig_output": rig_output,
    }


def _flatten_json(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested JSON into path -> leaf value pairs."""
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten_json(child, child_prefix))
        return flattened

    if isinstance(value, list):
        flattened: dict[str, Any] = {}
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            flattened.update(_flatten_json(child, child_prefix))
        return flattened

    return {prefix: value}


def _score_leaf_value(expected: Any, actual: Any) -> float:
    if expected is MISSING or actual is MISSING:
        return 0.0

    if type(expected) is not type(actual):
        return 0.0

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if expected == actual:
            return 1.0
        scale = max(abs(float(expected)), abs(float(actual)), 1.0)
        return max(0.0, 1.0 - (abs(float(expected) - float(actual)) / scale))

    return 1.0 if expected == actual else 0.0


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _top_level_section(path: str) -> str:
    for separator in (".", "["):
        if separator in path:
            return path.split(separator, 1)[0]
    return path or "root"


def _format_percent(score: float) -> float:
    return round(score * 100, 2)


def _comparison_feedback(
    section_scores: dict[str, float],
    missing_in_generated: list[str],
    extra_in_generated: list[str],
    worst_paths: list[dict[str, Any]],
) -> str:
    parts = []

    if missing_in_generated:
        parts.append(f"Missing {len(missing_in_generated)} reference fields")

    weak_sections = [name for name, score in section_scores.items() if score < 0.75]
    if weak_sections:
        parts.append("Weak sections: " + ", ".join(weak_sections[:3]))

    mismatches = [
        item["path"]
        for item in worst_paths
        if item["status"] in {"missing", "mismatch"}
    ]
    if mismatches:
        parts.append("Largest mismatches: " + ", ".join(mismatches[:3]))

    if extra_in_generated:
        parts.append(f"Generated {len(extra_in_generated)} extra fields")

    if not parts:
        return "Strong match across all scored JSON arguments."

    return ". ".join(parts) + "."


def compare_rig_json(expected_path: str, generated_path: str) -> dict[str, Any]:
    """Compare two rig JSON files path-by-path and score their similarity."""
    with open(expected_path, "r", encoding="utf-8") as f:
        expected = json.load(f)
    with open(generated_path, "r", encoding="utf-8") as f:
        generated = json.load(f)

    expected_flat = _flatten_json(expected)
    generated_flat = _flatten_json(generated)

    all_paths = sorted(set(expected_flat) | set(generated_flat))
    shared_paths = sorted(set(expected_flat) & set(generated_flat))

    path_scores = []
    missing_in_generated = []
    extra_in_generated = []
    section_buckets: dict[str, list[float]] = {}

    for path in all_paths:
        expected_value = expected_flat.get(path, MISSING)
        generated_value = generated_flat.get(path, MISSING)
        score = round(_score_leaf_value(expected_value, generated_value), 4)
        section = _top_level_section(path)
        section_buckets.setdefault(section, []).append(score)

        if expected_value is MISSING:
            extra_in_generated.append(path)
        elif generated_value is MISSING:
            missing_in_generated.append(path)

        path_scores.append(
            {
                "path": path,
                "score": score,
                "expected": None if expected_value is MISSING else expected_value,
                "generated": None if generated_value is MISSING else generated_value,
                "status": (
                    "extra"
                    if expected_value is MISSING
                    else "missing"
                    if generated_value is MISSING
                    else "match"
                    if score == 1.0
                    else "mismatch"
                ),
            }
        )

    shared_scores = [
        item["score"]
        for item in path_scores
        if item["path"] in shared_paths
    ]
    overall_scores = [item["score"] for item in path_scores]
    worst_paths = sorted(path_scores, key=lambda item: (item["score"], item["path"]))[:25]
    shared_score = round(_average(shared_scores), 4)
    overall_score = round(_average(overall_scores), 4)
    schema_coverage = round(
        (len(shared_paths) / len(expected_flat)) if expected_flat else 1.0,
        4,
    )
    section_scores = {
        section: round(_average(scores), 4)
        for section, scores in sorted(section_buckets.items())
    }
    feedback = _comparison_feedback(
        section_scores=section_scores,
        missing_in_generated=missing_in_generated,
        extra_in_generated=extra_in_generated,
        worst_paths=worst_paths,
    )

    return {
        "summary": {
            "expected_argument_count": len(expected_flat),
            "generated_argument_count": len(generated_flat),
            "shared_argument_count": len(shared_paths),
            "missing_argument_count": len(missing_in_generated),
            "extra_argument_count": len(extra_in_generated),
            "schema_coverage": schema_coverage,
            "shared_score": shared_score,
            "overall_score": overall_score,
            "numeric_score": _format_percent(overall_score),
            "feedback": feedback,
        },
        "section_scores": section_scores,
        "missing_in_generated": missing_in_generated,
        "extra_in_generated": extra_in_generated,
        "worst_paths": worst_paths,
        "path_scores": path_scores,
    }


def evaluate_all_images(
    images_dir: str,
    rigs_dir: str,
    output_dir: str,
    diecut_rounds: int = 1,
    diecut_model: str | None = None,
    vision_model: str | None = None,
    rig_model: str | None = None,
    use_genai_background: bool = False,
    tolerance: int = 30,
    fail_on_review: bool = False,
) -> dict[str, Any]:
    """Generate rigs for every image in a folder and compare them to reference rigs."""
    os.makedirs(output_dir, exist_ok=True)

    image_names = sorted(
        name
        for name in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, name))
        and os.path.splitext(name)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS
    )

    if not image_names:
        raise FileNotFoundError(f"No input images found in {images_dir}")

    results = []
    for image_name in image_names:
        stem, _ = os.path.splitext(image_name)
        image_path = os.path.join(images_dir, image_name)
        reference_rig_path = os.path.join(rigs_dir, f"{stem}.json")
        work_dir = os.path.join(output_dir, stem)

        result: dict[str, Any] = {
            "name": stem,
            "image_path": image_path,
            "reference_rig_path": reference_rig_path if os.path.exists(reference_rig_path) else None,
            "work_dir": work_dir,
        }

        try:
            outputs = run_pipeline(
                image_path=image_path,
                work_dir=work_dir,
                diecut_rounds=diecut_rounds,
                diecut_model=diecut_model,
                vision_model=vision_model,
                rig_model=rig_model,
                use_genai_background=use_genai_background,
                tolerance=tolerance,
                fail_on_review=fail_on_review,
            )
            result.update(outputs)

            if os.path.exists(reference_rig_path):
                comparison = compare_rig_json(reference_rig_path, outputs["rig_output"])
                result["status"] = "scored"
                result["comparison"] = comparison
                summary = comparison["summary"]
                print(
                    f"[score] {stem}: score={summary['numeric_score']:.2f}/100 "
                    f"coverage={summary['schema_coverage']:.4f} "
                    f"feedback={summary['feedback']}"
                )
            else:
                result["status"] = "missing_reference"
                print(f"[score] {stem}: generated rig, but no reference rig found at {reference_rig_path}")
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            print(f"[error] {stem}: {exc}")

        results.append(result)

    scored_results = [item for item in results if item.get("status") == "scored"]
    aggregate_section_scores: dict[str, list[float]] = {}
    for item in scored_results:
        for section, score in item["comparison"]["section_scores"].items():
            aggregate_section_scores.setdefault(section, []).append(score)

    aggregate = {
        "total_images": len(results),
        "scored_images": len(scored_results),
        "images_with_errors": sum(1 for item in results if item.get("status") == "error"),
        "images_missing_reference": sum(1 for item in results if item.get("status") == "missing_reference"),
        "average_overall_score": round(
            _average([item["comparison"]["summary"]["overall_score"] for item in scored_results]),
            4,
        ),
        "average_shared_score": round(
            _average([item["comparison"]["summary"]["shared_score"] for item in scored_results]),
            4,
        ),
        "average_schema_coverage": round(
            _average([item["comparison"]["summary"]["schema_coverage"] for item in scored_results]),
            4,
        ),
        "numeric_score": _format_percent(
            _average([item["comparison"]["summary"]["overall_score"] for item in scored_results])
        ),
        "section_scores": {
            section: round(_average(scores), 4)
            for section, scores in sorted(aggregate_section_scores.items())
        },
    }
    aggregate["feedback"] = _comparison_feedback(
        section_scores=aggregate["section_scores"],
        missing_in_generated=[],
        extra_in_generated=[],
        worst_paths=[],
    )

    report = {
        "images_dir": images_dir,
        "rigs_dir": rigs_dir,
        "output_dir": output_dir,
        "results": results,
        "aggregate": aggregate,
    }

    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[report] Wrote evaluation report to {report_path}")
    print(
        f"[report] Aggregate score={aggregate['numeric_score']:.2f}/100 "
        f"coverage={aggregate['average_schema_coverage']:.4f} "
        f"feedback={aggregate['feedback']}"
    )
    return report

def main():
    parser = argparse.ArgumentParser(description="Auto-rig CLI utilities.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["run", "evaluate-all"],
        default="run",
        help="CLI command to run",
    )
    parser.add_argument("--image", default="resources/hippo.png", help="Input image path")
    parser.add_argument("--diecut-output", default="diecut.png", help="Diecut output path")
    parser.add_argument("--pieces-dir", default="parts", help="Pieces output folder")
    parser.add_argument("--images-dir", default="resources/images", help="Folder of source images to evaluate")
    parser.add_argument("--rigs-dir", default="resources/rigs", help="Folder of reference rig JSON files")
    parser.add_argument("--output-dir", default="artifacts/rig_eval", help="Folder for generated rigs and reports")
    parser.add_argument("--diecut-rounds", type=int, default=1, help="Number of diecut refinement rounds")
    parser.add_argument("--diecut-model", default=None, help="Override the diecut generation model")
    parser.add_argument("--vision-model", default=None, help="Override the judge/bbox vision model")
    parser.add_argument("--rig-model", default=None, help="Override the rig parameter model")
    parser.add_argument(
        "--use-genai-background",
        action="store_true",
        help="Use GenAI background removal instead of the local remover",
    )
    parser.add_argument("--tolerance", type=int, default=30, help="Background removal tolerance")
    parser.add_argument(
        "--fail-on-review",
        action="store_true",
        help="Fail the diecut step if diecut review rejects the output",
    )
    args = parser.parse_args()

    if args.command == "evaluate-all":
        evaluate_all_images(
            images_dir=args.images_dir,
            rigs_dir=args.rigs_dir,
            output_dir=args.output_dir,
            diecut_rounds=args.diecut_rounds,
            diecut_model=args.diecut_model,
            vision_model=args.vision_model,
            rig_model=args.rig_model,
            use_genai_background=args.use_genai_background,
            tolerance=args.tolerance,
            fail_on_review=args.fail_on_review,
        )
        return

    agenticDiecut(args.image, args.diecut_output, args.pieces_dir)

if __name__ == "__main__":
    main()