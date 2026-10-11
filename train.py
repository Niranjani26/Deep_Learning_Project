import os
import time
import argparse
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import get_dataloaders
from models import get_model
from losses import DeblurLoss
from utils import calculate_psnr, calculate_ssim

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)

    for blur, sharp, _ in pbar:
        blur = blur.to(device, non_blocking=True)
        sharp = sharp.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with autocast():
            pred = model(blur)
            loss, loss_dict = criterion(pred, sharp)

        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "charb": f"{loss_dict.get('charbonnier', 0):.4f}"})

    return total_loss / len(loader)

def evaluate(model, loader, device, max_samples: int = 50):
    model.eval()
    psnr_scores = []
    ssim_scores = []

    with torch.no_grad():
        for i, (blur, sharp, _) in enumerate(loader):
            if i >= max_samples:
                break
            blur = blur.to(device)
            sharp = sharp.to(device)

            pred = model(blur)
            pred = torch.clamp(pred, 0.0, 1.0)

            psnr_scores.append(calculate_psnr(pred, sharp))
            ssim_scores.append(calculate_ssim(pred, sharp))

    avg_psnr = sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0.0
    avg_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0.0
    return avg_psnr, avg_ssim

def main():
    parser = argparse.ArgumentParser(description="Train Deblurring Model")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "srn", "restormer"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (4 recommended for 4GB VRAM)")
    parser.add_argument("--crop_size", type=int, default=128, help="Random crop size (128 or 256)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--val_samples", type=int, default=30, help="Number of validation images to test per epoch")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_loader, test_loader = get_dataloaders(
        root_dir="Gopro",
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        num_workers=0 # Set to 0 on Windows for maximum stability
    )

    model = get_model(args.model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = DeblurLoss().to(device) # Charbonnier + Edge Gradient loss for sharp restoration
    scaler = GradScaler()

    best_psnr = 0.0

    print(f"\n================ STARTING TRAINING (Epochs: {args.epochs}, Batch: {args.batch_size}, Crop: {args.crop_size}) ================")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        scheduler.step()

        val_psnr, val_ssim = evaluate(model, test_loader, device, max_samples=args.val_samples)
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Loss: {train_loss:.5f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | Time: {elapsed:.1f}s")

        # Save latest checkpoint
        latest_path = os.path.join(args.save_dir, f"latest_{args.model}.pth")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "psnr": val_psnr}, latest_path)

        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_path = os.path.join(args.save_dir, f"best_{args.model}.pth")
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "psnr": best_psnr}, best_path)
            print(f"  --> Saved new best checkpoint with PSNR: {best_psnr:.2f} dB")

    print(f"\nTraining completed! Best model saved to: {os.path.join(args.save_dir, f'best_{args.model}.pth')}")

if __name__ == "__main__":
    main()
