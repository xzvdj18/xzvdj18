import os
import uuid
import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# Binary mapping: 0 -> Normal, 1 -> Pneumonia
PNEUMONIA_CLASS_MAP = {
    0: "Normal",
    1: "Pneumonia"
}

# فحص مكان الموديل سواء كان بالمجلد الرئيسي أو داخل models/
CHEST_MODEL_PATH = 'chest_xray.keras' if os.path.exists('chest_xray.keras') else os.path.join(MODELS_FOLDER, 'chest_xray.keras')
YOLO_MODEL_PATH = 'best.pt' if os.path.exists('best.pt') else os.path.join(MODELS_FOLDER, 'best.pt')

# Flask initialized
app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')
CORS(app)  # تفعيل CORS للتواصل مع Netlify

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================
# MODEL LOADING & HELPERS
# ==========================================
chest_model = None
yolo_model = None

try:
    if os.path.exists(CHEST_MODEL_PATH):
        chest_model = tf.keras.models.load_model(CHEST_MODEL_PATH)
        print("Chest X-Ray model loaded successfully.")
    else:
        print(f"Warning: Chest model file not found at {CHEST_MODEL_PATH}")
except Exception as e:
    print(f"Error loading Chest X-Ray model: {str(e)}")

try:
    if os.path.exists(YOLO_MODEL_PATH):
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("YOLO Brain Tumor model loaded successfully.")
    else:
        print(f"Warning: YOLO model file not found at {YOLO_MODEL_PATH}")
except Exception as e:
    print(f"Error loading YOLO model: {str(e)}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_keras_input_shape(model):
    try:
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        h, w = input_shape[1], input_shape[2]
        if h is None or w is None:
            return (224, 224)
        return (int(h), int(w))
    except Exception:
        return (224, 224)


# ==========================================
# API ENDPOINTS
# ==========================================
@app.route('/')
def index():
    return jsonify({'status': 'Server is running', 'message': 'Medical AI Portal API active'})


@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/predict/pneumonia', methods=['POST'])
def predict_pneumonia():
    if chest_model is None:
        return jsonify({'error': 'Chest X-Ray model is not loaded on server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part provided in request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG, WEBP allowed.'}), 400

    try:
        target_size = get_keras_input_shape(chest_model)
        image = Image.open(file.stream).convert('RGB')
        image_resized = image.resize(target_size)
        
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        raw_pred = chest_model.predict(img_array)
        
        if raw_pred.shape[-1] == 1:
            score = float(raw_pred[0][0])
            predicted_class_idx = 1 if score >= 0.5 else 0
            confidence = score if predicted_class_idx == 1 else (1.0 - score)
        else:
            predicted_class_idx = int(np.argmax(raw_pred[0]))
            confidence = float(raw_pred[0][predicted_class_idx])

        label = PNEUMONIA_CLASS_MAP.get(predicted_class_idx, "Unknown")
        confidence_percentage = round(confidence * 100, 2)

        return jsonify({
            'success': True,
            'prediction': label,
            'confidence': confidence_percentage
        })

    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 500


@app.route('/predict/tumor', methods=['POST'])
def predict_tumor():
    if yolo_model is None:
        return jsonify({'error': 'YOLO Brain Tumor model is not loaded on server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part provided in request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG, WEBP allowed.'}), 400

    try:
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_id = str(uuid.uuid4())
        orig_filename = f"orig_{unique_id}.{ext}"
        orig_filepath = os.path.join(app.config['UPLOAD_FOLDER'], orig_filename)
        file.save(orig_filepath)

        results = yolo_model.predict(source=orig_filepath, conf=0.25)
        res = results[0]

        detections = []
        tumor_detected = False

        if len(res.boxes) > 0:
            tumor_detected = True
            for box in res.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0].cpu().numpy()) * 100
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = yolo_model.names.get(cls_id, "Tumor")

                detections.append({
                    'class_name': cls_name,
                    'confidence': round(conf, 2),
                    'box': coords
                })

        annotated_frame = res.plot()
        result_filename = f"res_{unique_id}.jpg"
        result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_filepath, annotated_frame)

        # رابط كامل للصورة ليعمل مع Netlify
        base_url = request.host_url.rstrip('/')

        return jsonify({
            'success': True,
            'tumor_detected': tumor_detected,
            'detections': detections,
            'result_image': f"{base_url}/uploads/{result_filename}",
            'original_image': f"{base_url}/uploads/{orig_filename}"
        })

    except Exception as e:
        return jsonify({'error': f'Failed to run detection: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)