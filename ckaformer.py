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
        return (X) / (X.norm(dim=-1, keepdim=True))


class LinearClassifier(nn.Module):
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
    def __init__(self, lcls, gamma=1.0):
        super().__init__()
        self.lcls = lcls
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = gamma

    def forward(self, X):
        P = self.lcls(X)
        P = self.softmax(P)

        derivative, _ = CKA_derivative(X, P)
        return X + self.gamma * derivative
        # return X + self.gamma * P @ P.T @ X - self.gamma * X @ X.T @ X


class CKAFormer(nn.Module):
    def __init__(self, dim, depth, out_dim):
        super().__init__()
        self.lcls = LinearClassifier(dim, out_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.Sequential(
                    LayerNorm(),
                    CompressionAnnilation(
                        self.lcls,
                        gamma=0.01,
                    ),
                )
            )

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X, self.lcls(X)
