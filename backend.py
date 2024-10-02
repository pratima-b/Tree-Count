from flask import Flask, request, jsonify
import os
import cv2
import base64
from io import BytesIO
from PIL import Image
from final_counting import ImageSeg  # Ensure this is the correct path to your ImageSeg class

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    try:
        # Process the image using ImageSeg class
        obj = ImageSeg(image_path)
        final_count = obj.count_trees()
        marked_img = obj.mark_trees()

        # Convert marked image to PNG and then to base64 string for transmission
        _, buffer = cv2.imencode('.png', marked_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Return the tree count and the processed image as base64 string
        return jsonify({
            'tree_count': final_count,
            'processed_image': img_base64
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
