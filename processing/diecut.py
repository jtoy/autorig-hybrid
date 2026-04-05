import io
import json
import os
from google.genai import types
from PIL import Image


JUDGE_CRITERIA = [
    ("part_count",
     "Are there exactly 14 separate body parts visible in the diecut image? "
     "Count each distinct piece carefully. The expected parts are: head, torso, "
     "right_upperarm, left_upperarm, right_forearm, left_forearm, right_hand, left_hand, "
     "right_thigh, left_thigh, right_calf, left_calf, right_foot, left_foot."),
    ("segmentation",
     "Are the arms properly split into upper arms, forearms, and hands as separate pieces? "
     "Are the legs properly split into thighs, calves, and feet as separate pieces? "
     "Each arm must be three distinct parts, each leg must be three distinct parts."),
    ("style_fidelity",
     "Does the diecut preserve the original art style, line weights, colors, "
     "and proportions? Is it a faithful extraction from the original rather than "
     "a redrawn or vectorized version?"),
    ("detail_preservation",
     "Are hands, feet, facial features, and small details unchanged from the original image? "
     "No added fingers, redrawn features, or modified details?"),
    ("layout",
     "Are all parts arranged horizontally in a row with clear spacing between them, "
     "not overlapping each other?"),
]

def diecut(client, imagePath, outputPath, fail_on_review: bool = False, rounds: int = 5, model: str = "gemini-3.1-flash-image-preview", judge_model: str = "gemini-3.1-flash-lite-preview"):
    """
    Performs character die-cutting for animation using Gemini.
    Separates head, torso, arms, hands, legs, and feet into a single image.

    Args:
        client: The Google GenAI client instance.
        imagePath: Path to the input image file.
        outputPath: Path where the resulting image will be saved.
        model: Model to use for diecut generation.
        judge_model: Model to use for judging diecut quality.
    """
    temperature = 0
    prompt = """
    EXTRACT the character into 14 separate pieces on white background.
    14 pieces: head, torso, 2 upper arms, 2 forearms, 2 hands, 2 thighs, 2 calves, 2 feet.

    CRITICAL:
    - Keep pieces in their ORIGINAL POSITIONS with small gaps at joints.
    - Each arm = 3 pieces (upperarm, forearm, hand). Cut at shoulder, elbow, wrist.
    - Each leg = 3 pieces (thigh, calf, foot). Cut at hip, knee, ankle.
    - Preserve original style, proportions, and details exactly.
    - Do NOT redraw hands or feet.
    - Layout: keep each part on the same side of the canvas as in the original (left-side parts toward the left, right-side parts toward the right) so position (x) consistently indicates which side each piece is on.

    Image 1 = example input. Image 2 = example output. Image 3 = process this.
    """

    if not os.path.exists(imagePath):
        raise FileNotFoundError(f"Image file not found: {imagePath}")

    # Load zero-shot examples
    example_input_path = os.path.join(os.path.dirname(__file__), "..", "resources", "tobyturtle.png")
    example_output_path = os.path.join(os.path.dirname(__file__), "..", "resources", "tobyturtle-diecut2.png")
    
    example_input = Image.open(example_input_path)

    example_output = Image.open(example_output_path)

    image = Image.open(imagePath)
    
    def pil_to_bytes(img):
        buffer = io.BytesIO()
        img.save(buffer, "PNG")
        return buffer.getvalue()

    def extract_inline_image(parts):
        for part in parts:
            if part.inline_data is not None:
                return part.inline_data.data, (part.inline_data.mime_type or "image/png")
        return None, None

    def judge_image(original_bytes, generated_bytes, generated_mime):
        """Score a diecut attempt across multiple binary criteria.

        Returns (scores_dict, total_score, feedback_str) where scores_dict
        maps criterion name to {"pass": bool, "reason": str}.
        """
        example_input_bytes = pil_to_bytes(example_input)
        example_output_bytes = pil_to_bytes(example_output)

        scores = {}
        for i, (name, criterion_prompt) in enumerate(JUDGE_CRITERIA, 1):
            prompt_text = (
                "You are judging a character diecut result. "
                "Image 1 is an example character. Image 2 is an example of a good diecut. "
                "Image 3 is the ORIGINAL character. Image 4 is the DIECUT to judge.\n\n"
                f"Criterion: {criterion_prompt}\n\n"
                'Respond with JSON: {"pass": true or false, "reason": "brief explanation"}'
            )
            try:
                print(f"[judge] Checking {i}/{len(JUDGE_CRITERIA)}: {name}...")
                resp = client.models.generate_content(
                    model=judge_model,
                    contents=[
                        types.Part.from_bytes(data=example_input_bytes, mime_type="image/png"),
                        types.Part.from_text(text="EXAMPLE CHARACTER"),
                        types.Part.from_bytes(data=example_output_bytes, mime_type="image/png"),
                        types.Part.from_text(text="EXAMPLE GOOD DIECUT"),
                        types.Part.from_bytes(data=original_bytes, mime_type="image/png"),
                        types.Part.from_text(text="ORIGINAL CHARACTER"),
                        types.Part.from_bytes(data=generated_bytes, mime_type=generated_mime or "image/png"),
                        types.Part.from_text(text="DIECUT TO JUDGE"),
                        types.Part.from_text(text=prompt_text),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                    ),
                )
                result = json.loads(resp.text.strip())
                scores[name] = {
                    "pass": bool(result.get("pass", False)),
                    "reason": result.get("reason", ""),
                }
            except Exception as e:
                scores[name] = {"pass": False, "reason": f"Judge error: {e}"}

        total = sum(1 for s in scores.values() if s["pass"])
        parts = [
            f"{name}:{'PASS' if s['pass'] else 'FAIL'}"
            for name, s in scores.items()
        ]
        feedback = f"{total}/5 — " + " | ".join(parts)
        return scores, total, feedback

    print(f"[diecut] Starting with model={model}, rounds={rounds}, image={imagePath}")

    chat = client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            temperature=temperature,
            image_config=types.ImageConfig(
                aspect_ratio="16:9",
                # image_size="1K",
            ),
        ),
    )

    round_prompts = max(1, rounds)
    generated_bytes = None
    generated_mime = None
    last_feedback = None
    attempts = []  # [(score, feedback, image_bytes, mime)]

    for round_index in range(1, round_prompts + 1):
        if round_index == 1:
            print(f"[diecut] Round {round_index}/{round_prompts}: generating diecut with {model}...")
            message_parts = [
                types.Part.from_text(text=prompt),
                types.Part.from_text(text="Example Input"),
                types.Part.from_bytes(data=pil_to_bytes(example_input), mime_type="image/png"),
                types.Part.from_text(text="Example Output"),
                types.Part.from_bytes(data=pil_to_bytes(example_output), mime_type="image/png"),
                types.Part.from_text(text="Now process this image"),
                types.Part.from_bytes(data=pil_to_bytes(image), mime_type="image/png"),
            ]
        else:
            if generated_bytes is None:
                raise ValueError("No generated image returned from previous round.")
            print(f"[diecut] Round {round_index}/{round_prompts}: retrying with feedback...")
            message_parts = [
                types.Part.from_text(text=last_feedback),
            ]

        response = chat.send_message(message_parts)
        generated_bytes, generated_mime = extract_inline_image(response.parts)
        if generated_bytes is None:
            raise ValueError("No image returned by model.")
        print(f"[diecut] Round {round_index}: got image, running judge ({judge_model})...")

        scores, total, feedback = judge_image(
            pil_to_bytes(image), generated_bytes, generated_mime
        )
        attempts.append((total, feedback, generated_bytes, generated_mime))
        print(f"[diecut] Judge (round {round_index}): {feedback}")

        if total == 5:
            print(f"Round {round_index} passed all criteria!")
            break

        # Build targeted feedback from only failed criteria
        failed_reasons = [
            f"- {s['reason']}" for s in scores.values() if not s["pass"]
        ]
        last_feedback = "Fix these issues:\n" + "\n".join(failed_reasons)

    # Select the best attempt (highest score; ties broken by earliest round)
    best = max(attempts, key=lambda a: a[0])
    best_score = best[0]
    best_bytes = best[2]
    best_round = attempts.index(best) + 1

    if best_score == 5:
        print(f"[diecut] Perfect score in round {best_round}!")
    else:
        print(f"[diecut] Best: round {best_round} ({best_score}/5 across {len(attempts)} attempt(s))")
        if fail_on_review:
            print("[diecut] fail_on_review=True, will raise after saving.")

    output_image = Image.open(io.BytesIO(best_bytes))
    output_image.save(outputPath)

    # Left/right are assigned by x-position in the bboxes step; no side_map needed.
    if best_score < 5 and fail_on_review:
        raise ValueError(
            f"Visual verification failed (best: {best_score}/5): {best[1]}"
        )

    return None
