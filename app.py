import os
import io
import base64
import time
import torch
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image

from models.restormer import Restormer
from utils import pad_image_for_model, unpad_image, tensor_to_pil, pil_to_tensor

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Flask App] Running on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

restormer_model = None

def load_transformer_model():
    global restormer_model
    if restormer_model is not None:
        return restormer_model

    print(f"Loading Restormer (Vision Transformer Encoder-Decoder) onto {device}...")
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    ).to(device)

    ckpt_path = os.path.join("checkpoints", "motion_deblurring.pth")
    if os.path.exists(ckpt_path):
        print(f"Loading SOTA Restormer weights from {ckpt_path}...")
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt["params"] if isinstance(ckpt, dict) and "params" in ckpt else ckpt
            model.load_state_dict(state_dict)
            print("Successfully loaded SOTA Restormer Transformer checkpoint (CVPR 2022 GoPro Benchmark)!")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print(f"Notice: Checkpoint not found at {ckpt_path}.")

    model.eval()
    restormer_model = model
    return restormer_model

@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "-1"
    return response

def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_residual_heatmap(orig_pil: Image.Image, rest_pil: Image.Image) -> str:
    orig_np = np.array(orig_pil, dtype=np.float32)
    rest_np = np.array(rest_pil, dtype=np.float32)
    diff = np.mean(np.abs(rest_np - orig_np), axis=2)

    diff_norm = np.clip(diff * 6.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heat_pil = Image.fromarray(heatmap_rgb)
    return pil_to_base64(heat_pil)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/restore", methods=["POST"])
def restore_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        input_pil = Image.open(file.stream).convert("RGB")
        w, h = input_pil.size

        # Scale to max dimension 800 for optimal balance between razor-sharp detail and fast GPU inference
        max_dim = 800
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            # Ensure divisible by 8
            new_w -= (new_w % 8)
            new_h -= (new_h % 8)
            input_pil = input_pil.resize((new_w, new_h), Image.BICUBIC)
            w, h = input_pil.size

        transformer = load_transformer_model()

        input_tensor = pil_to_tensor(input_pil).to(device)
        padded_tensor, pad_info = pad_image_for_model(input_tensor, divisor=8)

        start_time = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                restored_padded = transformer(padded_tensor)
                restored_tensor = unpad_image(restored_padded, pad_info)
                restored_tensor = torch.clamp(restored_tensor, 0.0, 1.0)
            if device.type == "cuda":
                torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        restored_pil = tensor_to_pil(restored_tensor)
        heatmap_b64 = generate_residual_heatmap(input_pil, restored_pil)

        return jsonify({
            "original": pil_to_base64(input_pil),
            "restored": pil_to_base64(restored_pil),
            "heatmap": heatmap_b64,
            "latency": f"{latency_ms:.0f} ms",
            "resolution": f"{w}x{h}",
            "device": str(device).upper(),
            "model": "RESTORMER (TRANSFORMER ENCODER-DECODER)"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/sample/<filename>")
def get_sample(filename):
    return send_from_directory("inputs", filename)

if __name__ == "__main__":
    load_transformer_model()
    app.run(host="127.0.0.1", port=5000, debug=False)
