import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from cka import CKA_derivative


class LayerNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X / (X.norm(dim=-1, keepdim=True))


class FeedForwardClassifier(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def forward(self, X):
        return self.linear(X)


class CompressionAnnilation(nn.Module):
    def __init__(self, lcls, layer=None, gamma=1e-4):
        super().__init__()
        self.lcls = lcls
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = gamma
        self.layer = layer

    def forward(self, X):
        P = self.lcls(X)
        P = self.softmax(P)

        # derivative, props = CKA_derivative(X, P)
        # return X + derivative, props
        props = {"lc": self.gamma, "rc": self.gamma}
        return X + self.gamma * P @ P.T @ X - self.gamma * X @ X.T @ X, props


class CKAFormer(nn.Module):
    def __init__(self, dim, depth, out_dim):
        super().__init__()
        self.lcls = FeedForwardClassifier(dim, out_dim)
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(
                nn.Sequential(
                    LayerNorm(),
                    CompressionAnnilation(
                        self.lcls,
                        layer=i,
                    ),
                )
            )
        self.stats = {}

    def forward(self, X):
        self.stats["lc"] = []
        self.stats["rc"] = []
        for layer in self.layers:
            X, prop = layer(X)
            self.stats["lc"].append(prop["lc"])
            self.stats["rc"].append(prop["rc"])
        return self.lcls(X), self.stats
