import tensorflow as tf
from flask import Flask, request, send_from_directory, render_template
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.models import load_model
from werkzeug.utils import secure_filename
import numpy as np
import os

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_SIZE = (160, 160)
UPLOAD_FOLDER = 'uploads'
modelo = load_model('modelos/catordog.h5')

def predict(file):
    test_image = image.load_img(file, target_size=(160, 160))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    predictions = modelo.predict(test_image).flatten()

    return predictions

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template('index.html', label = '', img = 'file://null')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    not_imagem = False
    if request.method == 'GET':
        return render_template('index.html')
    else:
        not_imagem = True
        file = request.files['file']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)

        result = predict(fullname)
        print(result)
        pred_prob = result.item()

        if pred_prob > .5:
            label = 'Cachorro'
            accuracy = round(pred_prob * 10, 2)
        else:
            label = 'Gato'
            accuracy = round((1 - pred_prob) * 10, 2)

        print(file.filename)
        return render_template('index.html', img=file.filename, label=label, accuracy=accuracy, not_imagem = not_imagem)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
