from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('skindiseases.h5')
target_img = os.path.join(os.getcwd(), 'static/images')


@app.route('/')
def index_view():
    return render_template('index.html')

ALLOWED_EXT = set(['jpg', 'jpeg', 'png'])

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXT

def read_image(filename):
    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path)
            class_prediction = model.predict(img)
            classes_x = np.argmax(class_prediction, axis=1)
            if classes_x == 0:
                disease = 'Acne and Rosacea Photos'
            elif classes_x == 1:
                disease = 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions'
            elif classes_x == 2:
                disease = 'Atopic Dermatitis Photos'
            elif classes_x == 3:
                disease = 'Bullous Disease Photos'
            elif classes_x == 4:
                disease = 'Cellulitis Impetigo and other Bacterial Infections'
            elif classes_x == 5:
                disease = 'Eczema Photos'
            elif classes_x == 6:
                disease = 'Exanthems and Drug Eruptions'
            elif classes_x == 7:
                disease = 'Hair Loss Photos Alopecia and other Hair Diseases'
            elif classes_x == 8:
                disease = 'Herpes HPV and other STDs Photos'
            elif classes_x == 9:
                disease = 'Light Diseases and Disorders of Pigmentation'
            elif classes_x == 10:
                disease = 'Lupus and other Connective Tissue diseases'
            elif classes_x == 11:
                disease = 'Melanoma Skin Cancer Nevi and Moles'
            elif classes_x == 12:
                disease = 'Nail Fungus and other Nail Disease'
            elif classes_x == 13:
                disease = 'Poison Ivy Photos and other Contact Dermatitis'
            elif classes_x == 14:
                disease = 'Psoriasis pictures Lichen Planus and related diseases'
            elif classes_x == 15:
                disease = 'Scabies Lyme Disease and other Infestations and Bites'
            elif classes_x == 16:
                disease = 'Seborrheic Keratoses and other Benign Tumors'
            elif classes_x == 17:
                disease = 'Systemic Disease'
            elif classes_x == 18:
                disease = 'Tinea Ringworm Candidiasis and other Fungal Infections'
            elif classes_x == 19:
                disease = 'Urticaria Hives'
            elif classes_x == 20:
                disease = 'Vascular Tumors'
            elif classes_x == 21:
                disease = 'Vasculitis Photos'
            elif classes_x == 22:
                disease = 'Warts Molluscum and other Viral Infections'
            else:
                disease = 'Unknown Disease'
            return render_template('predict.html', disease=disease, prob=class_prediction, user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)