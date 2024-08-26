from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'static'

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Process the image
            face_image_path = process_image(filepath)
            
            return render_template('index.html', original_image=file.filename, face_image=face_image_path)

    return render_template('index.html')

def process_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Get the first face detected
        (x, y, w, h) = faces[0]
        face_img = img[y:y+h, x:x+w]
        
        # Save the face image
        face_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'face_' + os.path.basename(image_path))
        cv2.imwrite(face_image_path, face_img)
        
        return 'face_' + os.path.basename(image_path)
    else:
        return 'default_face.png'  # Or handle cases where no face is detected

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
