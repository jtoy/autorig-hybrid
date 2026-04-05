from google.genai import types
from PIL import Image

def change_background(client, input_path, color, output_path):
    """
    Changes the background of an image using Google GenAI and saves it to output_path.
    """
    prompt = f"Keeping the same exact picture and resoluction change the backgound color to {color}."
    temperature = 0  
    model = "gemini-2.5-flash-image"

    print(f"Changing background to {color}...")
    image = Image.open(input_path)
    
    response = client.models.generate_content(
        model=model,
        contents=[prompt, image],
        config=types.GenerateContentConfig(
            temperature=temperature,
        ),
    )

    for part in response.parts:
        if part.inline_data is not None:
            img = part.as_image()
            img.save(output_path)
            return img
    
    return None