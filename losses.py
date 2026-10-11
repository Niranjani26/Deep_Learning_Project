import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (Smoothed L1):
    L(x, y) = sqrt((x - y)^2 + eps^2)
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + (self.eps * self.eps)))
        return loss

class EdgeLoss(nn.Module):
    """
    Edge/Gradient Loss using Sobel operators:
    Enforces sharp structural boundaries and prevents blurred or smoothed edges.
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        # Sobel convolution kernels for horizontal and vertical gradient
        k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("kernel_x", k_x.repeat(3, 1, 1, 1))
        self.register_buffer("kernel_y", k_y.repeat(3, 1, 1, 1))

    def _gradient(self, x: torch.Tensor):
        grad_x = F.conv2d(x, self.kernel_x, padding=1, groups=3)
        grad_y = F.conv2d(x, self.kernel_y, padding=1, groups=3)
        return grad_x, grad_y

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_gx, pred_gy = self._gradient(pred)
        target_gx, target_gy = self._gradient(target)

        diff_x = pred_gx - target_gx
        diff_y = pred_gy - target_gy

        loss = torch.mean(torch.sqrt(diff_x * diff_x + diff_y * diff_y + (self.eps * self.eps)))
        return loss

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual Loss using intermediate feature representations of VGG-19.
    """
    def __init__(self, feature_layer: int = 16):
        super().__init__()
        try:
            weights = models.VGG19_Weights.DEFAULT
            vgg = models.vgg19(weights=weights).features[:feature_layer].eval()
        except Exception:
            vgg = models.vgg19(pretrained=True).features[:feature_layer].eval()

        for param in vgg.parameters():
            param.requires_grad = False
            
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std

        feat_pred = self.vgg(pred_norm)
        feat_target = self.vgg(target_norm)
        return F.l1_loss(feat_pred, feat_target)

class DeblurLoss(nn.Module):
    """
    Composite CNN Deblurring Loss:
    L_total = L_charbonnier + 0.05 * L_edge
    """
    def __init__(self, charbonnier_weight: float = 1.0, edge_weight: float = 0.05):
        super().__init__()
        self.charbonnier_loss = CharbonnierLoss()
        self.edge_loss = EdgeLoss()
        self.charbonnier_weight = charbonnier_weight
        self.edge_weight = edge_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        charb = self.charbonnier_loss(pred, target)
        edge = self.edge_loss(pred, target)

        total = (self.charbonnier_weight * charb) + (self.edge_weight * edge)

        losses = {
            "charbonnier": charb.item(),
            "edge": edge.item(),
            "total": total.item()
        }
        return total, losses

if __name__ == "__main__":
    criterion = DeblurLoss()
    x = torch.rand(2, 3, 128, 128)
    y = torch.rand(2, 3, 128, 128)
    loss, l_dict = criterion(x, y)
    print("CNN DeblurLoss test passed! Loss values:", l_dict)
