"""Small convolutional AE and a diagonal Gaussian feature-distance baseline."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class FeatureModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:6])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        return self.features((x - self.mean) / self.std)


def anomaly_maps(autoencoder, cnn, x, feature_mean, feature_variance):
    ae_map = (autoencoder(x) - x).square().mean(1)
    feature_map = ((cnn(x) - feature_mean).square() / feature_variance).mean(1)
    feature_map = F.interpolate(
        feature_map[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False
    )[:, 0]
    return ae_map, feature_map


def image_score(anomaly_map):
    """Top 1% mean emphasises local defects; fixed before test evaluation."""
    flat = anomaly_map.flatten(1)
    return flat.topk(max(1, round(flat.shape[1] * 0.01)), dim=1).values.mean(1)
