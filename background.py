from google import genai
from processing import change_background

def main():
    client = genai.Client()
    image_path = "diecut.png"
    output_path = "diecut_black.png"

    change_background(client, image_path, "black", output_path)

if __name__ == "__main__":
    main()