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


class Compression(nn.Module):
    def __init__(self, dim, num_classes, layer=None, gamma=1e-4, trainable_mean=False):
        super().__init__()
        self.layer = layer

        if not trainable_mean:
            self.register_buffer("weighted_means", torch.zeros(num_classes, dim))
        else:
            self.weighted_means = nn.Linear(num_classes, dim, bias=False)

        self.gamma = gamma
        self.num_classes = num_classes

        self.softmax = nn.Softmax(dim=-1)
        self.fn = nn.Linear(dim, num_classes)

        self.alpha = 0.9
        self.trainable_mean = trainable_mean

        self.init = False

    def forward(self, X):

        P = self.softmax(self.fn(X))
        if self.train and not self.trainable_mean:
            current_mean = (P.T @ X).detach() / X.shape[0]
            self.weighted_means = self.weighted_means * (self.alpha) + current_mean * (
                1 - self.alpha
            )
        elif self.train and not self.init:
            # self.weighted_means.weight = (P.T @ X).detach() / X.shape[0]
            self.weighted_means.weight = nn.Parameter(
                (P.T @ X).detach() / X.shape[0], requires_grad=True
            )
            self.init = True

        if self.trainable_mean:
            W = self.weighted_means.weight
        else:
            W = self.weighted_means

        X = X + self.gamma * P @ W

        return X


class Annihilation(nn.Module):
    def __init__(self, layer=None, gamma=1e-4):
        super().__init__()
        self.layer = layer

        self.alpha = 0.9

        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("running_cov", torch.zeros(1))

    def forward(self, X):
        props = {"lc": self.gamma, "rc": self.gamma}

        if self.training:
            W = X.T @ X / X.shape[0]
            if self.running_cov.shape[0] == 1:
                self.running_cov = W.clone().detach()
            self.running_cov = (
                self.alpha * self.running_cov + (1 - self.alpha) * W.detach()
            )

        X = X - self.gamma * X @ self.running_cov
        return X, props


class CKAFormer(nn.Module):
    def __init__(
        self, dim, depth, out_dim, num_classes, trainable_mean=False, gamma=1e-4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(
                nn.Sequential(
                    LayerNorm(),
                    Compression(
                        num_classes=num_classes,
                        dim=dim,
                        layer=i,
                        gamma=gamma,
                        trainable_mean=trainable_mean,
                    ),
                    Annihilation(
                        layer=i,
                        gamma=gamma,
                    ),
                )
            )
        self.layers.append(FeedForwardClassifier(dim, out_dim))
        self.stats = {}

    def forward(self, X):
        self.stats["lc"] = []
        self.stats["rc"] = []
        for layer in self.layers[:-1]:
            X, prop = layer(X)
            self.stats["lc"].append(prop["lc"])
            self.stats["rc"].append(prop["rc"])
        return self.layers[-1](X), self.stats
