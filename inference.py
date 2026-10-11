import os
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from models.unet import UNetDeblur
from utils import pad_image_for_model, unpad_image, tensor_to_pil, pil_to_tensor, calculate_psnr, calculate_ssim

def run_inference(image_path: str, checkpoint_path: str = "checkpoints/best_unet.pth", ground_truth_path: str = None, output_dir: str = "outputs", device_str: str = "auto"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Determine device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Initialize CNN model
    model = UNetDeblur(base_channels=32).to(device)
    model.eval()

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading trained CNN weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        if isinstance(checkpoint, dict) and "psnr" in checkpoint:
            print(f"Checkpoint PSNR: {checkpoint['psnr']:.2f} dB (Epoch: {checkpoint.get('epoch', 'N/A')})")
    else:
        print(f"Note: No checkpoint found at {checkpoint_path}. Running with initialized weights.")

    # Load input image
    input_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = input_pil.size
    print(f"Processing image: {image_path} (Resolution: {orig_w}x{orig_h})")

    input_tensor = pil_to_tensor(input_pil).to(device)

    # Pad image dynamically so dimensions are divisible by 16
    padded_tensor, pad_info = pad_image_for_model(input_tensor, divisor=16)

    with torch.no_grad():
        restored_padded = model(padded_tensor)
        restored_tensor = unpad_image(restored_padded, pad_info)
        restored_tensor = torch.clamp(restored_tensor, 0.0, 1.0)

    restored_pil = tensor_to_pil(restored_tensor)

    # Check metrics if ground truth is available
    metrics_str = ""
    if ground_truth_path and os.path.exists(ground_truth_path):
        gt_pil = Image.open(ground_truth_path).convert("RGB")
        if gt_pil.size != input_pil.size:
            gt_pil = gt_pil.resize(input_pil.size, Image.BICUBIC)
        gt_tensor = pil_to_tensor(gt_pil).to(device)
        psnr_val = calculate_psnr(restored_tensor, gt_tensor)
        ssim_val = calculate_ssim(restored_tensor, gt_tensor)
        metrics_str = f" | PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}"
        print(f"Evaluation against Ground Truth:{metrics_str}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_restored_path = os.path.join(output_dir, f"{base_name}_restored.png")
    out_comp_path = os.path.join(output_dir, f"{base_name}_comparison.png")

    restored_pil.save(out_restored_path)

    # Build side-by-side comparison canvas
    comp_img = Image.new("RGB", (orig_w * 2, orig_h + 40), color=(18, 22, 32))
    comp_img.paste(input_pil, (0, 40))
    comp_img.paste(restored_pil, (orig_w, 40))

    # Add text labels
    draw = ImageDraw.Draw(comp_img)
    draw.text((20, 12), "ORIGINAL (BLURRY INPUT)", fill=(248, 113, 113))
    draw.text((orig_w + 20, 12), f"RESTORED / AUTO-COMPLETED (CNN){metrics_str}", fill=(74, 222, 128))

    comp_img.save(out_comp_path)
    print(f"Saved sharp image: {out_restored_path}")
    print(f"Saved side-by-side comparison: {out_comp_path}")
    return out_restored_path, out_comp_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Image Deblurring Inference")
    parser.add_argument("--image", type=str, default="inputs/sample_user_upload.png", help="Path to input blurry image")
    parser.add_argument("--sharp", type=str, default=None, help="Optional ground truth sharp image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_unet.pth", help="Path to trained model weights (.pth)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    run_inference(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        ground_truth_path=args.sharp,
        output_dir=args.output_dir
    )
