import os
import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = './data'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model_path = "./mnist_model.keras"
model = load_model(model_path, safe_mode=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    image_resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR)
    image_processed = 255 - image_resized
    image_input = image_processed.astype("float32").reshape(1, 28, 28, 1) / 255.0

    predictions = model.predict(image_input)[0]  # (10,)
    top_indices = predictions.argsort()[-10:][::-1]  # Top 5
    top_predictions = [(int(i), float(predictions[i]) * 100) for i in top_indices]

    # Lưu ảnh dưới dạng CSV
    image_flatten = image_processed.flatten().reshape(1, -1)
    columns = [str(i) for i in range(784)]
    df = pd.DataFrame(image_flatten, columns=columns)
    csv_path = "./pixels_28x28.csv"
    df.to_csv(csv_path, index=False)

    return top_predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Không có file được chọn')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('Không có file được chọn')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        uploaded_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if uploaded_image is None:
            flash('Không thể đọc ảnh vừa tải lên. Vui lòng thử lại.')
            return redirect(url_for('index'))

        top_predictions = process_image(file_path)

        if top_predictions is None:
            flash('Không thể xử lý ảnh. Vui lòng thử lại.')
            return redirect(url_for('index'))

        return render_template('index.html',
                               filename=filename,
                               predictions=top_predictions)

    flash('Định dạng file không được hỗ trợ. Vui lòng chọn file PNG, JPG hoặc JPEG.')
    return redirect(url_for('index'))

@app.route('/data/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
