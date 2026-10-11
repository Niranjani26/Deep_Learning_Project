import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention:
    Dynamically recalibrates channel-wise feature responses by explicitly
    modelling interdependencies between channels.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """
    Residual Block with Channel Attention (RCAB).
    Combines convolution, instance normalization, and channel attention.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True)
        )
        self.ca = ChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv(x)
        res = self.ca(res)
        return x + res

class DilatedResidualBlock(nn.Module):
    """
    Dilated Residual Block for wide receptive field to capture large motion blur trajectories.
    """
    def __init__(self, channels: int, dilation: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True)
        )
        self.ca = ChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ca(self.conv(x))

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNetDeblur(nn.Module):
    """
    Attention-Enhanced CNN Encoder-Decoder (U-Net) for Blind Image Deblurring.
    Architecture Features:
    1. Hierarchical Multi-Scale Encoder-Decoder.
    2. Residual Channel Attention Blocks (RCAB) at every stage.
    3. Dilated Bottleneck Convolutions for wide motion blur receptive field.
    4. Multi-Level Skip Connections for high-frequency edge propagation.
    5. Global Residual Learning (Output = Blurry Input + Learned Sharp Residual).
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels

        # 1. Shallow Feature Extraction
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 2. Hierarchical Encoder
        self.enc1 = nn.Sequential(DoubleConv(c, c), ResidualBlock(c))
        self.down1 = nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1)

        self.enc2 = nn.Sequential(DoubleConv(c * 2, c * 2), ResidualBlock(c * 2))
        self.down2 = nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1)

        self.enc3 = nn.Sequential(DoubleConv(c * 4, c * 4), ResidualBlock(c * 4))
        self.down3 = nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=2, padding=1)

        # 3. Dilated Receptive Field Bottleneck
        self.bottleneck = nn.Sequential(
            DilatedResidualBlock(c * 8, dilation=2),
            DilatedResidualBlock(c * 8, dilation=4),
            ResidualBlock(c * 8)
        )

        # 4. Decoder with Skip Connections
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c * 8, c * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec3 = nn.Sequential(DoubleConv(c * 8, c * 4), ResidualBlock(c * 4))

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c * 4, c * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec2 = nn.Sequential(DoubleConv(c * 4, c * 2), ResidualBlock(c * 2))

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c * 2, c, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec1 = nn.Sequential(DoubleConv(c * 2, c), ResidualBlock(c))

        # 5. Output Reconstruction Tail
        self.tail = nn.Conv2d(c, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shallow features
        f_in = self.head(x)

        # Encoder path
        e1 = self.enc1(f_in)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        # Bottleneck path
        b = self.bottleneck(self.down3(e3))

        # Decoder path with skip connections
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # High-frequency residual correction
        residual = self.tail(d1)

        # Global Residual Learning: Sharp Image = Blurry Input + Residual
        out = x + residual
        return out

if __name__ == "__main__":
    model = UNetDeblur(base_channels=32)
    dummy_input = torch.rand(1, 3, 256, 256)
    out = model(dummy_input)
    print("Enhanced CNN Deblur Model forward pass successful! Output:", out.shape)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable CNN Parameters: {num_params:,}")
