import os
import uuid
from flask import Flask, render_template, request, url_for, redirect, flash
from werkzeug.utils import secure_filename
import numpy as np
import tifffile as tiff
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "unet_resnet34_3bands_best.h5")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
ALLOWED_EXT = {"tif", "tiff"}
TARGET_SIZE = (128, 128)
THRESH = 0.5

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "replace-with-a-secret-key"

# ---------------- Helpers ----------------
def allowed_file(fname):
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def ensure_channel_layout(img):
    """Return array with shape (C,H,W) float32 expecting C==12"""
    arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[2] == 12:
            return np.transpose(arr, (2,0,1)).astype(np.float32)
        if arr.shape[0] == 12:
            return arr.astype(np.float32)
    raise ValueError(f"Unsupported image shape {arr.shape}. Expected 12 channels in (H,W,12) or (12,H,W).")

def resize_channels(c_h_w, size=TARGET_SIZE):
    C, H, W = c_h_w.shape
    if (H, W) == size:
        return c_h_w
    resized = np.zeros((C, size[1], size[0]), dtype=np.float32)
    for i in range(C):
        resized[i] = cv2.resize(c_h_w[i].astype(np.float32), (size[0], size[1]), interpolation=cv2.INTER_LINEAR)
    return resized

def standardize_c_h_w(c_h_w):
    out = np.zeros_like(c_h_w, dtype=np.float32)
    for i in range(c_h_w.shape[0]):
        band = c_h_w[i]
        m = band.mean()
        s = band.std()
        out[i] = (band - m) / s if s > 0 else band
    return out

def prepare_input_from_path(path, channel_indices=[4,5,6]):
    raw = tiff.imread(path)
    c_h_w = ensure_channel_layout(raw)            
    c_h_w = resize_channels(c_h_w, size=TARGET_SIZE)
    c_h_w_std = standardize_c_h_w(c_h_w)
    sel = c_h_w_std[channel_indices, :, :]        
    hwc = np.transpose(sel, (1,2,0)).astype(np.float32)
    return np.expand_dims(hwc, axis=0), hwc      

def hwc_visual(hwc):
    vis = np.zeros_like(hwc, dtype=np.uint8)
    for i in range(3):
        band = hwc[:,:,i]
        lo, hi = np.percentile(band, (2,98))
        if hi - lo <= 0:
            scaled = np.clip(band, 0, 255)
        else:
            scaled = (band - lo) / (hi - lo) * 255.0
        vis[:,:,i] = np.clip(scaled, 0, 255).astype(np.uint8)
    return vis

def overlay_and_save(vis_rgb_uint8, pred_prob, outpath):
    overlay = vis_rgb_uint8.copy()
    if pred_prob.min() < 0 or pred_prob.max() > 1:
        pred_norm = (pred_prob - pred_prob.min())/(pred_prob.max()-pred_prob.min()+1e-9)
    else:
        pred_norm = pred_prob
    alpha = 0.6
    red = overlay[:,:,0].astype(np.float32)
    red = (1-alpha)*red + alpha*(pred_norm*255)
    overlay[:,:,0] = np.clip(red,0,255).astype(np.uint8)
    Image.fromarray(overlay).save(outpath)

# ---------------- Load model ----------------
print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded.")

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No file field named 'image' in request.")
        return redirect(request.url)
    file = request.files["image"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(request.url)
    if not allowed_file(file.filename):
        flash("Unsupported file type. Use .tif or .tiff")
        return redirect(request.url)

    fname = secure_filename(file.filename)
    uid = uuid.uuid4().hex
    save_name = f"{uid}_{fname}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], save_name)
    file.save(save_path)

    try:
        inp, hwc = prepare_input_from_path(save_path, channel_indices=[4,5,6])
        pred = model.predict(inp)[0]
        if pred.ndim == 3:
            pred = np.squeeze(pred, axis=-1)
        bin_mask = (pred >= THRESH).astype(np.uint8) * 255

        # save visuals
        vis = hwc_visual(hwc)
        vis_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_vis.png")
        Image.fromarray(vis).save(vis_path)

        prob_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_prob.png")
        plt.imsave(prob_path, pred, cmap="viridis")

        bin_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_mask.png")
        Image.fromarray(bin_mask).convert("L").save(bin_path)

        overlay_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_overlay.png")
        overlay_and_save(vis, pred, overlay_path)

        try:
            os.remove(save_path)
        except Exception:
            pass

        return render_template("index.html",
                               predicted_image=url_for('static', filename=f"uploads/{os.path.basename(overlay_path)}"),
                               debug_files={
                                   "Input visual": url_for('static', filename=f"uploads/{os.path.basename(vis_path)}"),
                                   "Probability map": url_for('static', filename=f"uploads/{os.path.basename(prob_path)}"),
                                   "Binary mask": url_for('static', filename=f"uploads/{os.path.basename(bin_path)}")
                               })

    except Exception as e:
        flash(f"Processing error: {e}")
        return redirect(url_for("index"))

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
