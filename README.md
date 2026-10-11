# Deep Learning-Based Image Deblurring & Auto-Completion Studio

A deep learning framework for blind image deblurring, dynamic scene restoration, and high-frequency edge completion on the GoPro benchmark dataset.

---

## 🏛️ Implemented Architectures

This project implements and compares three major deep learning paradigms:

1. **U-Net Baseline (CNN / Autoencoder)**:
   - Symmetrical Encoder-Decoder with multi-scale skip connections.
   - Dual residual convolution blocks per stage.
   - Global residual learning ($I_{pred} = I_{blur} + \Delta$).
2. **Scale-Recurrent Network (SRN / ConvLSTM / RNN)**:
   - Multi-scale Gaussian pyramid decomposition ($1/4 \to 1/2 \to 1/1$).
   - Convolutional LSTM cell maintaining spatial recurrent memory across scales.
   - Progressive coarse-to-fine sharpness refinement.
3. **Restormer (Vision Transformer / Attention)**:
   - Multi-Dconv Head Transposed Self-Attention (MDTA) operating across channels.
   - Gated-Dconv Feed-Forward Network (GDFN).
   - Long-range contextual restoration without spatial quadratic complexity.

---

## 🚀 How to Upload & Verify Any Image

### Option 1: Interactive Browser Web Studio
Launch the local web app with drag-and-drop file upload and interactive before/after split slider:
```bash
py -3.11 app.py
```
Open **http://127.0.0.1:5000** in your browser. Drag and drop any photo and click **Auto-Complete & Deblur**.

### Option 2: Command-Line Single Image Inference
Place your photo in `inputs/` and run:
```bash
py -3.11 inference.py --image inputs/sample_user_upload.png --model unet
```
The result and side-by-side comparison are saved directly to `outputs/`.

---

## 🏋️ Training the Models

To train on the GoPro dataset using Automatic Mixed Precision (AMP) on your NVIDIA GPU:

### 1. Train U-Net (CNN)
```bash
py -3.11 train.py --model unet --epochs 20 --batch_size 4 --crop_size 128
```

### 2. Train Scale-Recurrent ConvLSTM (RNN)
```bash
py -3.11 train.py --model srn --epochs 20 --batch_size 4 --crop_size 128
```

### 3. Train Restormer (Transformer)
```bash
py -3.11 train.py --model restormer --epochs 20 --batch_size 4 --crop_size 128
```

Saved checkpoints are stored in `checkpoints/best_<model>.pth`.

---

## 📊 Comparative Evaluation & Benchmarking

To benchmark all models across PSNR (dB), SSIM, Latency (ms), and Parameter count:
```bash
py -3.11 evaluate.py --models unet srn restormer --samples 30
```

---

## 📁 Repository Structure

```text
PROJECT_MAIN/
├── Gopro/                   # GoPro dataset (train/test with blur and sharp pairs)
├── inputs/                  # Folder to drop your custom uploaded images
├── outputs/                 # Output directory for restored images and comparisons
├── checkpoints/             # Saved model weights (.pth)
├── models/
│   ├── __init__.py          # Factory function get_model()
│   ├── unet.py              # CNN Encoder-Decoder
│   ├── recurrent_srn.py     # Scale-Recurrent ConvLSTM (RNN)
│   └── restormer.py         # Restormer Vision Transformer
├── dataset.py               # Paired GoPro DataLoader with patch cropping & augmentations
├── losses.py                # Charbonnier Loss & Perceptual VGG Loss
├── utils.py                 # PSNR, SSIM, dynamic padding/unpadding
├── train.py                 # Training pipeline with AMP (FP16)
├── evaluate.py              # Comparative benchmark script
├── inference.py             # Single-image inference script
├── app.py                   # Flask Web UI server
└── templates/
    └── index.html           # Dark-glassmorphism Web UI
```
