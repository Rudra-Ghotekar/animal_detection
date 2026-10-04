from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json


MODEL_PATH = "animal_classifier_efficientnet.h5"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)
LOW_CONF_THRESHOLD = 25.0
MID_CONF_THRESHOLD = 75.0

print("Loading EfficientNet model...")
eff_model = load_model(MODEL_PATH)

print("Loading VGG16 model (ImageNet pre-trained)...")
vgg_model = VGG16(weights='imagenet')

if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r") as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = [
        'bear', 'boar', 'cat', 'chimpanzee', 'cow', 'deer', 'dog', 'donkey',
        'elephant', 'fox', 'goat', 'gorilla', 'hippopotamus', 'horse', 'leopard',
        'lion', 'mouse', 'orangutan', 'ox', 'panda', 'pig', 'rat', 'reindeer',
        'rhinoceros', 'sheep', 'tiger', 'wolf', 'zebra'
    ]

print(f"Loaded {len(CLASS_NAMES)} animal classes.")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        file = request.files.get("file")
        if file:

            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            img = image.load_img(file_path, target_size=IMG_SIZE)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

            preds = eff_model.predict(img_array)
            eff_conf = round(100 * np.max(preds), 2)
            eff_pred_index = int(np.argmax(preds))
            eff_pred_class = CLASS_NAMES[eff_pred_index] if eff_pred_index < len(CLASS_NAMES) else "Unknown"

            if eff_conf < LOW_CONF_THRESHOLD:
                final_label = "Animal not present"
                confidence = eff_conf
            elif LOW_CONF_THRESHOLD <= eff_conf < MID_CONF_THRESHOLD:
                print("Using VGG16 secondary detection...")
                pil_img = Image.open(file_path).convert("RGB").resize((224, 224))
                vgg_img = np.expand_dims(image.img_to_array(pil_img), axis=0)
                vgg_img = preprocess_input(vgg_img)
                vgg_preds = vgg_model.predict(vgg_img)
                decoded = decode_predictions(vgg_preds, top=1)[0][0]
                final_label = decoded[1].replace("_", " ")
                confidence = round(100 * decoded[2], 2)
            else:
                final_label = eff_pred_class
                confidence = eff_conf

            return render_template(
                "index.html",
                prediction=final_label,
                confidence=confidence,
                img_path=file_path
            )

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
