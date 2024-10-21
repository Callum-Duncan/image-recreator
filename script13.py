import os
from tkinter import Tk, filedialog
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange
import scipy

# larger immage widths require a lot more memory and processing time

# 8 gigs can run:           # suggested upper limit but can defetily do more
# desighed width: 40 * 2000
# max threads: not recomended to use all with 8 threads use 5
# ---
# output image width will resize the output image, this lowwers resolution so increase with the desighed width
# 20,000 is a good high

Config = {
    "Images": "imgs", # the folder with all the images used to recreate the input
    "desired_width": 32*300, # width of the image in 
                # images width * how many wide

    "part_size": 32, # resize images
    "max_threads": 5,
    "Output_image_width": 5000, # size of the recreated output image
}

def load_part_images(folder_path):
    return [os.path.join(root, file)
            for root, _, files in os.walk(folder_path)
            for file in files
            if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

@njit
def calculate_average_rgb(image, area=3):
    total = np.zeros(3, dtype=np.float64)
    height, width, _ = image.shape

    # Define the middle 50% region      # only slightly faster 'remove?'
    start_row = height // area          # lowering the area decrease color accuracy
    end_row = 3 * height // area
    start_col = width // area
    end_col = 3 * width // area

    # only the middle area of the image
    for i in range(start_row, end_row):
        for j in range(start_col, end_col):
            total += image[i, j, :3]

    # Calculate the number of pixels in the middle areaa
    num_pixels = (end_row - start_row) * (end_col - start_col)

    # Calculate the average RGB values
    average_rgb = total / num_pixels

    return average_rgb

@njit
def calculate_squared_distance(color1: np.ndarray, color2: np.ndarray):
    distance = 0.0
    n = len(color1)
    for i in prange(n):  # Use prange for parallel execution
        diff = color1[i] - color2[i]
        distance += diff * diff  # Squared difference
    return distance
    # return Numpy.linalg.norm(color1-color2)
    # return cdist(color1, color2, metric='euclidean')

@njit
def find_closest_matching_part(input_rgb: np.ndarray, avg_colors: np.ndarray) -> int:
    closest_index = -1
    min_distance = 1e10  # large constant instead of np.inf

    n_avg_colors = avg_colors.shape[0]  # Store the number of average colors


    # Iterate over all average colors to find the closest match
    for i in range(n_avg_colors):
        # Calculate squared distance using the helper function
        distance = calculate_squared_distance(avg_colors[i], input_rgb)
        
        # print(distance)
        # Update closest color if this one is closer

        if distance < min_distance:
            min_distance = distance
            closest_index = i

    return closest_index
    # colors = avg_colors
    # color = input_rgb
    # distances = np.sqrt(np.sum((colors-color)**2,axis=1))
    # return np.where(distances==np.amin(distances))

@njit
def resize_image_maintain_aspect_ratio(input_image, desired_width):
    height, width = input_image.shape[:2]
    new_height = int(height * (desired_width / width))
    resized_image = np.zeros((new_height, desired_width, 3), dtype=input_image.dtype)

    for i in range(new_height):
        for j in range(desired_width):
            orig_x = j * width // desired_width
            orig_y = i * height // new_height
            pixel = input_image[orig_y, orig_x]

            # Remove alpha if present
            if input_image.shape[-1] == 4:
                alpha = pixel[3] / 255.0
                resized_image[i, j] = (pixel[:3] * alpha).astype(np.uint8)
            else:
                resized_image[i, j] = pixel

    return resized_image

def process_image(part_image):
    if part_image is None:
        return None, None

    part_image_resized = resize_image_maintain_aspect_ratio(part_image, Config["part_size"])
    avg_rgb = calculate_average_rgb(part_image_resized)
    return avg_rgb, part_image_resized

def precalculate_part_info(part_images):
    avg_colors = []
    resized_images = []

    images_in_memory = [cv2.imread(image_path, cv2.IMREAD_UNCHANGED) for image_path in part_images]
    del part_images

    max_workers = min(Config["max_threads"], len(images_in_memory))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_image, images_in_memory), desc="Precalculating Part Info", total=len(images_in_memory), ascii=' ▖▘▝▗▚▞█', maxinterval=0.3))
        
    del images_in_memory

    results = [result for result in results if result is not None]
    if results:
        avg_colors, resized_images = zip(*results)
        avg_colors = np.array(avg_colors)
        resized_images = list(resized_images)

    return avg_colors, resized_images

def create_matching_image(input_image, avg_colors, resized_images):
    # Resize the input image to the desired width
    input_image_resized = resize_image_maintain_aspect_ratio(input_image, Config["desired_width"])
    del input_image  # Free memory

    height, width, _ = input_image_resized.shape
    part_size = Config["part_size"]
    cols = width // part_size
    rows = height // part_size

    output_image = np.zeros((height, width, 3), dtype=np.uint8)

    coordinates = [(x * part_size, y * part_size) for y in range(rows) for x in range(cols)]
    
    regions = [
        input_image_resized[y:y + part_size, x:x + part_size, :3] for x, y in tqdm(coordinates, desc="Extracting Regions", ascii=True)
    ]

    del input_image_resized

    average_rgbs = [
        calculate_average_rgb(region)
        for region in tqdm(regions, desc="Calculating average rgbs", ascii=True)
    ]

    del regions

    closest_indices = [
        find_closest_matching_part(avg_rgb, avg_colors) 
        for avg_rgb in tqdm(average_rgbs, desc="Finding Closest Matching Parts", ascii=True)
    ]
    del average_rgbs

    with tqdm(total=len(coordinates), desc="Placing Closest Parts", ascii=True, mininterval=1) as pbar:
        for (x, y), closest_idx in zip(coordinates, closest_indices):
            output_image[y:y + part_size, x:x + part_size] = resized_images[closest_idx]
            pbar.update(1)

    output_image = resize_image_maintain_aspect_ratio(output_image, Config["Output_image_width"])
    output_image_path = os.path.join("output", "output_img.png")
    cv2.imwrite(output_image_path, output_image)

    return output_image_path

def process_images_and_create_processed_images(input_image_path, avg_colors, resized_images):
    input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    processed_image_path = create_matching_image(input_image, avg_colors, resized_images)
    del input_image
    print(f"Processed image saved at: {processed_image_path}")

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    input_image_path = filedialog.askopenfilename(title="Select the input image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if input_image_path:
        part_images = load_part_images(Config["Images"])
        if part_images:
            avg_colors, resized_images = precalculate_part_info(part_images)
            process_images_and_create_processed_images(input_image_path, avg_colors, resized_images)
        else:
            print("No part images found in the specified folder. Exiting.")
    else:
        print("No input image selected. Exiting.")




# improve average rgbs ^
# old: 854,789 : 8s

# improve reigons
# old: 2,535,002 : 2s

# improve closes matching part ^^
# old: 166,199 : 40s