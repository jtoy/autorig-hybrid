from google import genai
from processing import bboxes

def main():
    client = genai.Client()
    image_path = "diecut.png"
    output_dir = "parts"

    bboxes(client, image_path, output_dir)

if __name__ == "__main__":
    main()