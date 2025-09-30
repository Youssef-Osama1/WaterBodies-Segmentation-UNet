import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tifffile as tiff
import cv2

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

MODEL_PATH = r"C:\Users\Youssef\Downloads\Computer Vision\WaterBodies-Segmentation-UNet\unet_resnet34_3bands_best.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(image_path):
    img = tiff.imread(image_path).astype(np.float32)
    img = cv2.resize(img, (128, 128))
    img = img[:, :, [4, 5, 6]]

    for i in range(img.shape[2]):
        band = img[:, :, i]
        mean, std = np.mean(band), np.std(band)
        if std > 0:
            img[:, :, i] = (band - mean) / std

    img = np.expand_dims(img, axis=0)

    return img


def predict_mask(image_array):
    pred = model.predict(image_array)
    pred_bin = (pred > 0.5).astype(np.uint8)
    return pred_bin[0, :, :, 0]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file_img = request.files.get("image")

        if file_img and file_img.filename != '':
            filename_img = secure_filename(file_img.filename)
            filepath_img = os.path.join(app.config["UPLOAD_FOLDER"], filename_img)
            file_img.save(filepath_img)

            img_array = preprocess_image(filepath_img)
            pred_mask = predict_mask(img_array)

            pred_filename = "pred_" + os.path.splitext(filename_img)[0] + ".png"
            pred_path = os.path.join(app.config["UPLOAD_FOLDER"], pred_filename)
            cv2.imwrite(pred_path, pred_mask * 255)

            return render_template(
                "index.html",
                predicted_image=pred_path
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
