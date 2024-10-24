import os
from flask import Flask, render_template, request, jsonify, url_for
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

Config = {
    "Images": "imgs",
    "desired_width": 32*200,
    "part_size": 32,
    "max_threads": 5,
    "Output_image_width": 5000,
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

    start_row = height // area
    end_row = 3 * height // area
    start_col = width // area
    end_col = 3 * width // area

    for i in range(start_row, end_row):
        for j in range(start_col, end_col):
            total += image[i, j, :3]

    num_pixels = (end_row - start_row) * (end_col - start_col)
    average_rgb = total / num_pixels

    return average_rgb

@njit
def calculate_squared_distance(color1: np.ndarray, color2: np.ndarray):
    distance = 0.0
    n = len(color1)
    for i in prange(n):
        diff = color1[i] - color2[i]
        distance += diff * diff
    return distance

@njit
def find_closest_matching_part(input_rgb: np.ndarray, avg_colors: np.ndarray) -> int:
    closest_index = -1
    min_distance = 1e10

    n_avg_colors = avg_colors.shape[0]

    for i in range(n_avg_colors):
        distance = calculate_squared_distance(avg_colors[i], input_rgb)
        if distance < min_distance:
            min_distance = distance
            closest_index = i

    return closest_index

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
    input_image_resized = resize_image_maintain_aspect_ratio(input_image, Config["desired_width"])
    del input_image

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
    output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], "output_img.png")
    
    # Delete old output image if it exists
    if os.path.exists(output_image_path):
        os.remove(output_image_path)
    
    cv2.imwrite(output_image_path, output_image)

    return output_image_path

def process_images_and_create_processed_images(input_image_path, avg_colors, resized_images):
    input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    processed_image_path = create_matching_image(input_image, avg_colors, resized_images)
    del input_image
    print(f"Processed image saved at: {processed_image_path}")
    return processed_image_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file:
            # Read slider values from the request
            Config["desired_width"] = int(request.form.get('hidden_desired_width', Config["desired_width"]))
            Config["Output_image_width"] = int(request.form.get('hidden_output_image_width', Config["Output_image_width"]))
            Config["part_size"] = int(request.form.get('hidden_part_size', Config["part_size"]))
            Config["max_threads"] = int(request.form.get('hidden_max_threads', Config["max_threads"]))

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            part_images = load_part_images(Config["Images"])
            avg_colors, resized_images = precalculate_part_info(part_images)
            processed_image_path = process_images_and_create_processed_images(filepath, avg_colors, resized_images)
            return jsonify({
                "input_image": url_for('static', filename=f'uploads/{file.filename}'),
                "output_image": url_for('static', filename='output/output_img.png')
            })
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)