from PIL import Image


# Read image as
def image_to_list(image):
    img = Image.open(image).convert('L')  # convert image to 8-bit grayscale
    WIDTH, HEIGHT = img.size

    data = list(img.getdata())  # convert image data to a list of integers
    # convert that to 2D list (list of lists of integers)
    data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]

    return flatten_and_normalize(data)


# Flatten double list and set pixel values between 1-0 from 255-0
def flatten_and_normalize(data):
    flat = []
    # flatten image in single list and set values between 1-0
    for row in data:
        for value in row:
            flat.append(float(value / 255.0))
    return flat
