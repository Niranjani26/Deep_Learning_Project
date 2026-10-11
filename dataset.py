import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class GoProDataset(Dataset):
    """
    Paired GoPro Deblurring Dataset.
    Expects directory structure:
        root_dir/
            train/
                blur/
                sharp/
            test/
                blur/
                sharp/
    """
    def __init__(self, root_dir: str, split: str = "train", crop_size: int = 256, is_train: bool = True):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.crop_size = crop_size
        self.is_train = is_train

        self.blur_dir = os.path.join(root_dir, split, "blur")
        self.sharp_dir = os.path.join(root_dir, split, "sharp")

        if not os.path.exists(self.blur_dir) or not os.path.exists(self.sharp_dir):
            raise FileNotFoundError(f"Missing blur or sharp directory in: {os.path.join(root_dir, split)}")

        # Collect image filenames
        blur_files = set(os.listdir(self.blur_dir))
        sharp_files = set(os.listdir(self.sharp_dir))
        
        # Only take filenames present in both
        common_files = sorted(list(blur_files.intersection(sharp_files)))
        self.filenames = [f for f in common_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"[{split.upper()}] Loaded {len(self.filenames)} paired images from {root_dir}")

    def __len__(self):
        return len(self.filenames)

    def _paired_random_crop(self, blur_img: Image.Image, sharp_img: Image.Image):
        w, h = blur_img.size
        cs = self.crop_size

        if w < cs or h < cs:
            # Resize if smaller than crop size
            scale = max(cs / w, cs / h)
            new_w, new_h = int(w * scale) + 1, int(h * scale) + 1
            blur_img = blur_img.resize((new_w, new_h), Image.BICUBIC)
            sharp_img = sharp_img.resize((new_w, new_h), Image.BICUBIC)
            w, h = blur_img.size

        x = random.randint(0, w - cs)
        y = random.randint(0, h - cs)

        blur_crop = blur_img.crop((x, y, x + cs, y + cs))
        sharp_crop = sharp_img.crop((x, y, x + cs, y + cs))

        return blur_crop, sharp_crop

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        blur_path = os.path.join(self.blur_dir, filename)
        sharp_path = os.path.join(self.sharp_dir, filename)

        blur_img = Image.open(blur_path).convert("RGB")
        sharp_img = Image.open(sharp_path).convert("RGB")

        if self.is_train:
            # Random paired spatial crop
            blur_img, sharp_img = self._paired_random_crop(blur_img, sharp_img)

            # Random horizontal flip
            if random.random() > 0.5:
                blur_img = blur_img.transpose(Image.FLIP_LEFT_RIGHT)
                sharp_img = sharp_img.transpose(Image.FLIP_LEFT_RIGHT)

            # Random vertical flip
            if random.random() > 0.5:
                blur_img = blur_img.transpose(Image.FLIP_TOP_BOTTOM)
                sharp_img = sharp_img.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            # During evaluation, center crop or keep test patch size
            if self.crop_size is not None and self.crop_size > 0:
                w, h = blur_img.size
                cs = self.crop_size
                x = max(0, (w - cs) // 2)
                y = max(0, (h - cs) // 2)
                blur_img = blur_img.crop((x, y, min(w, x + cs), min(h, y + cs)))
                sharp_img = sharp_img.crop((x, y, min(w, x + cs), min(h, y + cs)))

        # Convert to torch tensor [C, H, W] normalized to [0, 1]
        blur_tensor = torch.from_numpy(np.array(blur_img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        sharp_tensor = torch.from_numpy(np.array(sharp_img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        return blur_tensor, sharp_tensor, filename

def get_dataloaders(root_dir: str = "Gopro", batch_size: int = 4, crop_size: int = 256, num_workers: int = 2):
    train_dataset = GoProDataset(root_dir=root_dir, split="train", crop_size=crop_size, is_train=True)
    test_dataset = GoProDataset(root_dir=root_dir, split="test", crop_size=crop_size, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader

if __name__ == "__main__":
    loader, _ = get_dataloaders(root_dir="Gopro", batch_size=2, crop_size=128, num_workers=0)
    for blur, sharp, names in loader:
        print(f"Batch loaded successfully! Blur: {blur.shape}, Sharp: {sharp.shape}, Sample: {names[0]}")
        break
