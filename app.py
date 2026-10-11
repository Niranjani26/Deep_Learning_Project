import os
import io
import base64
import time
import torch
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image

from models.unet import UNetDeblur
from utils import pad_image_for_model, unpad_image, tensor_to_pil, pil_to_tensor

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Flask App] Running on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

model = None
model_mtime = None

def load_cnn_model():
    global model, model_mtime
    ckpt_path = os.path.join("checkpoints", "best_unet.pth")
    current_mtime = os.path.getmtime(ckpt_path) if os.path.exists(ckpt_path) else None

    if model is not None and model_mtime == current_mtime:
        return model

    print(f"Loading CNN (UNetDeblur) onto {device}...")
    net = UNetDeblur(base_channels=32).to(device)
    net.eval()

    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            net.load_state_dict(state_dict)
            psnr_val = ckpt.get("psnr", "N/A") if isinstance(ckpt, dict) else "N/A"
            epoch_val = ckpt.get("epoch", "N/A") if isinstance(ckpt, dict) else "N/A"
            print(f"Loaded trained CNN! (Epoch: {epoch_val}, PSNR: {psnr_val:.2f} dB)")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print(f"Notice: No checkpoint found at {ckpt_path}.")

    model = net
    model_mtime = current_mtime
    return model

def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_residual_heatmap(orig_pil: Image.Image, rest_pil: Image.Image) -> str:
    # Compute amplified difference map
    orig_np = np.array(orig_pil, dtype=np.float32)
    rest_np = np.array(rest_pil, dtype=np.float32)
    diff = np.mean(np.abs(rest_np - orig_np), axis=2) # [H, W]

    # Amplify differences for clear visualization
    diff_norm = np.clip(diff * 8.0, 0, 255).astype(np.uint8)
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

        if max(w, h) > 1280:
            scale = 1280.0 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            new_w -= (new_w % 2)
            new_h -= (new_h % 2)
            input_pil = input_pil.resize((new_w, new_h), Image.BICUBIC)
            w, h = input_pil.size

        cnn = load_cnn_model()

        input_tensor = pil_to_tensor(input_pil).to(device)
        padded_tensor, pad_info = pad_image_for_model(input_tensor, divisor=16)

        start_time = time.perf_counter()
        with torch.no_grad():
            restored_padded = cnn(padded_tensor)
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
            "latency": f"{latency_ms:.1f} ms",
            "resolution": f"{w}x{h}",
            "device": str(device).upper()
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/sample/<filename>")
def get_sample(filename):
    return send_from_directory("inputs", filename)

if __name__ == "__main__":
    load_cnn_model()
    app.run(host="127.0.0.1", port=5000, debug=False)
