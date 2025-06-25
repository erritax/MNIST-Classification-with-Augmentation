import random
from io import BytesIO
from PIL import Image
from google import genai
from google.genai.types import GenerationConfig
from image_grid import collect_valid_images, create_grid

# configure gemini
client = genai.Client(api_key = "ENTER KEY")

# get random selection of images
digit = 0
count = 10
max_attempts = 10000
images = collect_valid_images(digit, count, max_attempts)

# generate grid of images
if len(images) == 10:
    grid_img = create_grid(images, 2, 5)

    # convert to png bytes
    rgb_grid = grid_img.convert("RGB")
    buffer = BytesIO()
    rgb_grid.save(buffer, format="PNG")
    buffer.seek(0)
    image_bytes = buffer.read()

    # this prompt can be altered based on augmentation style
    prompt = (f"These are 10 samples of the handwritten digit '{digit}' from the MNIST dataset. "
        f"Please generate one additional image of the digit '{digit}', potentially transforming"
        f"via geometric, elastic, and color space."
    )

    # prompt gemini with image and prompt, retrieve image and text output
    response = client.models.generate_content(
        model='gemini-2.0-flash-preview-image-generation',
        contents=[
            {"role": "user", "parts": [
                {"text": prompt},
                {"inline_data": {
                    "mime_type": "image/png",
                    "data": image_bytes
                }}
            ]}
        ],
        config=genai.types.GenerateContentConfig(
            responseModalities=['TEXT', 'IMAGE']
        )
    )

    # image and text output
    for part in response.candidates[0].content.parts:

        if hasattr(part, 'inline_data') and part.inline_data:
            img = Image.open(BytesIO(part.inline_data.data))
            img = img.resize((28, 28))  # force output to 28x28
            img.show()

        elif hasattr(part, 'text') and part.text:
            print("Text response:", part.text)
else:
    print("Not enough valid images collected.")