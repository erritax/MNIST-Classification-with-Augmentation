import random
import requests
from io import BytesIO
from PIL import Image

# to generate the url of an image given digit and image number
def generate_mnist_url(digit, number):
    base_url = "https://cumberland.isis.vanderbilt.edu/errita/mnist_png_dataset/test"
    return f"{base_url}/{digit}/{number}.png"

# to get "count" valid images
def collect_valid_images(digit, count, max_attempts):
    images = []
    tried = set()
    attempts = 0

    while (len(images) < count) and (attempts < max_attempts):
        # get random 
        num = random.randint(0, 10000)

        # skip duplicates
        if num in tried:
            continue

        tried.add(num)
        url = generate_mnist_url(digit, num)    # make url

        try:
            response = requests.get(url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert("L").resize((28, 28))
                images.append(img)
                print(f"Fetched: {url}")
            
            # invalid image number
            else:
                print(f"Missing: {url}")
                
        # invalid url
        except Exception as e:
            print(f"Error fetching {url}: {e}")

        attempts += 1

    return images

# to create a 2x5 grid of images
def create_grid(images, rows, cols):
    if len(images) != rows * cols:
        raise ValueError("Grid size doesn't match number of images")
    w, h = images[0].size
    grid = Image.new('L', (cols * w, rows * h))
    for i, img in enumerate(images):
        x = (i % cols) * w
        y = (i // cols) * h
        grid.paste(img, (x, y))
    return grid

# ====================

if __name__ == "__main__":
    # digit in question
    digit = 0
    # number of images
    count = 10
    # max number of attempts (safety from infinite loop)
    max_attempts = 10000
    
    images = collect_valid_images(digit, count, max_attempts)

    # show image 
    if len(images) == 10:
        grid_img = create_grid(images, rows = 2, cols = 5)
        grid_img.show()
    else:
        print("invalid")
