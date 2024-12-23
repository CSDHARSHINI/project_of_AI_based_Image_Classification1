import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from transformers import AutoImageProcessor, TFAutoModelForImageClassification
from PIL import Image
import logging
from werkzeug.utils import secure_filename
import tensorflow as tf
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure application with absolute paths
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
TEMPLATE_FOLDER = BASE_DIR / 'templates'

# Create necessary directories
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
TEMPLATE_FOLDER.mkdir(parents=True, exist_ok=True)

# Configure Flask app
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize model and processor
try:
    logger.info("Loading model and processor...")
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = TFAutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    logger.info("Model and processor loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        
        # If user does not select file
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            
            # Create full file path using Path
            filepath = UPLOAD_FOLDER / filename
            
            # Log the file path
            logger.info(f"Attempting to save file to: {filepath}")
            
            # Save the file
            file.save(str(filepath))
            logger.info(f"File saved successfully at: {filepath}")

            # Process the image
            try:
                with Image.open(filepath) as image:
                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Process image
                    inputs = processor(images=image, return_tensors="tf")
                    outputs = model(**inputs)
                    
                    # Get prediction
                    predicted_class_idx = int(tf.argmax(outputs.logits, axis=-1)[0])
                    description = model.config.id2label[predicted_class_idx]
                    confidence = float(tf.nn.softmax(outputs.logits, axis=-1)[0][predicted_class_idx])
                    
                    logger.info(f"Successfully classified image as: {description}")
                    
                    return jsonify({
                        'success': True,
                        'description': description,
                        'confidence': f"{confidence:.2%}",
                        'image_path': f'/uploads/{filename}'
                    })

            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return jsonify({'error': f'Error processing image: {str(e)}'}), 500

        else:
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload a JPG, JPEG, or PNG file'}), 400

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    # Reduce TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Print directory information for debugging
    logger.info(f"Base Directory: {BASE_DIR}")
    logger.info(f"Upload Directory: {UPLOAD_FOLDER}")
    logger.info(f"Template Directory: {TEMPLATE_FOLDER}")
    
    # Run the app
    app.run(debug=True, port=5000)