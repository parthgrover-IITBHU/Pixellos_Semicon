
import torch
import torch.nn as nn
import torch.nn.functional as F

def sobel_gradient(x):
    sobel_x = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
        device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
        device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 3)

    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt):
        return torch.mean(torch.sqrt((pred - gt) ** 2 + self.eps ** 2))


class RestorationLoss(nn.Module):
    # This matches the active loss in the supplied notebook:
    # Charbonnier pixel loss + Sobel edge L1 loss.
    def __init__(self, w_edge=1):
        super().__init__()
        self.w_edge = w_edge
        self.charbonnier = CharbonnierLoss()

    def forward(self, pred, gt):
        l_pixel = self.charbonnier(pred, gt)
        l_edge = F.l1_loss(sobel_gradient(pred), sobel_gradient(gt))
        total = l_pixel + self.w_edge * l_edge

        return total, {
            "pixel": l_pixel.item(),
            "edge": l_edge.item(),
        }
