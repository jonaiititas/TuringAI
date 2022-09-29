from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash
import cv2
import numpy as np
from joblib import load
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import os

UPLOAD_FOLDER = '../Images'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'super secret key'

# mtcnn
def extract_face(image, required_size=(160, 160)):
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = cv2.resize(image, required_size)
    face_array = np.asarray(image)
    return face_array


# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding

    yhat = model.predict(samples)

    return yhat[0]

def load_models():
    svc = load("models/svc.joblib")
    label_enc = load("models/label_enc.joblib")
    facenet = load_model('models/facenet_keras.h5')
    return svc, label_enc, facenet

def predict(img, facenet, svc, enc):
    aligned = extract_face(img)
    emb = get_embedding(facenet, aligned)
    samples = np.expand_dims(emb, axis=0)
    yhat_class = svc.predict(samples)
    yhat_prob = svc.predict_proba(samples)
    # get name
    class_index = yhat_class[0]
    class_probability = yhat_prob[0, class_index] * 100
    predict_names = enc.inverse_transform(yhat_class)
    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))

@app.route('/')
def index():
    return render_template('index.html', title='Home', user="User")

@app.route('/', methods=['POST'])
def upload_file():
    import io
    bio = io.BytesIO()
    print(request.files)
    request.files['image'].save(bio)
    img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
    # cv2.imwrite("Images/01.jpg", img)
    print('Loading models...')
    svc, enc, fn = load_models()
    print("Loaded models, predicting...")
    predict(img, fn, svc, enc)
    return render_template('index.html', title='Home', user="User")



if __name__ == "__main__":
    app.run(debug=True, threaded=False)