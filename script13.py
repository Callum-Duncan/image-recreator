from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

Config = {
    "Images": "example images",
    "desired_width": 32*300,
    "part_size": 32,
    "max_threads": 5,
    "Output_image_width": 5000,
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    example_images = os.listdir(Config["Images"])
    return render_template('index.html', config=Config, example_images=example_images)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return redirect(url_for('process_image', filename=filename))
    return redirect(request.url)

@app.route('/process/<filename>')
def process_image(filename):
    input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    part_images = load_part_images(Config["Images"])
    if part_images:
        avg_colors, resized_images = precalculate_part_info(part_images)
        processed_image_path = create_matching_image(input_image_path, avg_colors, resized_images)
        return render_template('index.html', input_image=filename, output_image=os.path.basename(processed_image_path), config=Config)
    else:
        return "No part images found in the specified folder. Exiting."

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True)