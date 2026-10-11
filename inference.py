import os
import argparse
import torch
from PIL import Image, ImageDraw
import numpy as np

from models.restormer import Restormer
from utils import pad_image_for_model, unpad_image, tensor_to_pil, pil_to_tensor, calculate_psnr, calculate_ssim

def run_inference(image_path: str, checkpoint_path: str = "checkpoints/motion_deblurring.pth", ground_truth_path: str = None, output_dir: str = "outputs", device_str: str = "auto"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model = Restormer().to(device)
    model.eval()

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading trained Transformer weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("Restormer SOTA weights loaded successfully!")
    else:
        print(f"Note: Checkpoint not found at {checkpoint_path}.")

    input_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = input_pil.size
    print(f"Processing image: {image_path} (Resolution: {orig_w}x{orig_h})")

    # Scale if larger than 800 for optimal memory & speed
    max_dim = 800
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        new_w -= (new_w % 8)
        new_h -= (new_h % 8)
        input_pil = input_pil.resize((new_w, new_h), Image.BICUBIC)
        w, h = input_pil.size
    else:
        w, h = orig_w, orig_h

    input_tensor = pil_to_tensor(input_pil).to(device)
    padded_tensor, pad_info = pad_image_for_model(input_tensor, divisor=8)

    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            restored_padded = model(padded_tensor)
            restored_tensor = unpad_image(restored_padded, pad_info)
            restored_tensor = torch.clamp(restored_tensor, 0.0, 1.0)

    restored_pil = tensor_to_pil(restored_tensor)

    metrics_str = ""
    if ground_truth_path and os.path.exists(ground_truth_path):
        gt_pil = Image.open(ground_truth_path).convert("RGB").resize((w, h), Image.BICUBIC)
        gt_tensor = pil_to_tensor(gt_pil).to(device)
        psnr_val = calculate_psnr(restored_tensor, gt_tensor)
        ssim_val = calculate_ssim(restored_tensor, gt_tensor)
        metrics_str = f" | PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}"
        print(f"Evaluation against Ground Truth:{metrics_str}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_restored_path = os.path.join(output_dir, f"{base_name}_restored.png")
    out_comp_path = os.path.join(output_dir, f"{base_name}_comparison.png")

    restored_pil.save(out_restored_path)

    comp_img = Image.new("RGB", (w * 2, h + 40), color=(250, 249, 253))
    comp_img.paste(input_pil, (0, 40))
    comp_img.paste(restored_pil, (w, 40))

    draw = ImageDraw.Draw(comp_img)
    draw.text((20, 12), "ORIGINAL (BLURRY INPUT)", fill=(225, 29, 72))
    draw.text((w + 20, 12), f"RESTORED (RESTORMER TRANSFORMER){metrics_str}", fill=(5, 150, 105))

    comp_img.save(out_comp_path)
    print(f"Saved sharp image: {out_restored_path}")
    print(f"Saved side-by-side comparison: {out_comp_path}")
    return out_restored_path, out_comp_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restormer Image Deblurring Inference")
    parser.add_argument("--image", type=str, default="inputs/sample_heavy_blur_0.png", help="Path to input blurry image")
    parser.add_argument("--sharp", type=str, default=None, help="Optional ground truth sharp image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/motion_deblurring.pth", help="Path to trained model weights (.pth)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    run_inference(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        ground_truth_path=args.sharp,
        output_dir=args.output_dir
    )
