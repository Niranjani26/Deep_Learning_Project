import os
import time
import argparse
import torch
from tabulate import tabulate

from dataset import get_dataloaders
from models import get_model
from utils import calculate_psnr, calculate_ssim

def evaluate_model(model_name: str, checkpoint_path: str = None, num_samples: int = 50, device_str: str = "cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    _, test_loader = get_dataloaders(root_dir="Gopro", batch_size=1, crop_size=256, num_workers=0)

    model = get_model(model_name).to(device)
    model.eval()

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded weights for {model_name} from: {checkpoint_path}")
    else:
        print(f"Warning: No weights found for {model_name}. Evaluating baseline initialization.")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    psnr_list = []
    ssim_list = []
    latencies = []

    with torch.no_grad():
        for i, (blur, sharp, _) in enumerate(test_loader):
            if i >= num_samples:
                break
            blur = blur.to(device)
            sharp = sharp.to(device)

            start = time.perf_counter()
            pred = model(blur)
            pred = torch.clamp(pred, 0.0, 1.0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)

            psnr_list.append(calculate_psnr(pred, sharp))
            ssim_list.append(calculate_ssim(pred, sharp))

    return {
        "Model": model_name.upper(),
        "Parameters": f"{num_params:,}",
        "PSNR (dB)": f"{sum(psnr_list)/len(psnr_list):.2f}",
        "SSIM": f"{sum(ssim_list)/len(ssim_list):.4f}",
        "Latency (ms)": f"{sum(latencies)/len(latencies):.2f} ms"
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Deblurring Models")
    parser.add_argument("--models", nargs="+", default=["unet", "srn", "restormer"], help="List of models to benchmark")
    parser.add_argument("--samples", type=int, default=20, help="Number of test images to evaluate")
    args = parser.parse_args()

    results = []
    for m in args.models:
        ckpt_path = os.path.join("checkpoints", f"best_{m}.pth")
        res = evaluate_model(m, checkpoint_path=ckpt_path, num_samples=args.samples)
        results.append(res)

    print("\n" + "=" * 60)
    print("           MODEL BENCHMARK & COMPARISON TABLE")
    print("=" * 60)
    try:
        from tabulate import tabulate
        print(tabulate(results, headers="keys", tablefmt="fancy_grid"))
    except ImportError:
        # Fallback if tabulate not installed
        for r in results:
            print(r)

if __name__ == "__main__":
    main()
