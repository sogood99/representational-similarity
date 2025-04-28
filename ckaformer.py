import torch
from torch import nn


class FeedForwardClassifier(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def forward(self, X):
        return self.linear(X.detach())


class CKAFormerBlock(nn.Module):
    def __init__(self, dim, num_classes, gamma=1e-4, trainable_mean=False, layer=None):
        super().__init__()

        self.gamma = gamma
        self.bn = nn.BatchNorm1d(dim, affine=False)
        self.compression = Compression(
            dim=dim,
            num_classes=num_classes,
            trainable_mean=trainable_mean,
            layer=layer,
        )
        self.annihilation = Annihilation(layer=layer)

    def forward(self, X):
        X_normed = self.bn(X)
        classifier_logits, X_compression = self.compression(X_normed)
        X_annihilation = self.annihilation(X_normed)
        return classifier_logits, X_normed + self.gamma * (X_compression - X_annihilation)


class Compression(nn.Module):
    def __init__(self, dim, num_classes, layer=None, trainable_mean=False):
        super().__init__()
        self.layer = layer

        if not trainable_mean:
            self.register_buffer("weighted_means", torch.zeros(num_classes, dim))
        else:
            self.weighted_means = nn.Linear(num_classes, dim, bias=False)

        self.num_classes = num_classes

        self.softmax = nn.Softmax(dim=-1)
        self.fn = nn.Linear(dim, num_classes)

        self.alpha = 0.99
        self.trainable_mean = trainable_mean

        self.init = False

    def forward(self, X):

        local_logits = self.fn(X.detach())
        P = self.softmax(local_logits)
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

        X = P @ W

        return local_logits, X


class Annihilation(nn.Module):
    def __init__(self, layer=None):
        super().__init__()
        self.layer = layer

        self.alpha = 0.99

        self.register_buffer("running_cov", torch.zeros(1))

    def forward(self, X):
        if self.training:
            W = X.T @ X / X.shape[0]
            if self.running_cov.shape[0] == 1:
                self.running_cov = W.clone().detach()
            self.running_cov = (
                    self.alpha * self.running_cov + (1 - self.alpha) * W.detach()
            )

        X = X @ self.running_cov
        return X


class CKAFormer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            out_dim,
            num_classes,
            trainable_mean=False,
            gamma=1e-4,
            save_hidden=False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for i in range(depth):
            self.blocks.append(
                CKAFormerBlock(
                    dim=dim,
                    num_classes=num_classes,
                    gamma=gamma,
                    trainable_mean=trainable_mean,
                    layer=i,
                )
            )
        self.blocks.append(FeedForwardClassifier(dim, out_dim))
        self.save_hidden = save_hidden

    def forward(self, X):
        stats = {}
        if self.save_hidden:
            stats["hidden"] = []

        all_logits = []
        for layer in self.blocks[:-1]:
            logits, X = layer(X)

            all_logits.append(logits)

            if self.save_hidden:
                stats["hidden"].append(X.clone().detach())

        output_logits = self.blocks[-1](X)
        if self.save_hidden:
            stats["hidden"].append(torch.log_softmax(output_logits, dim=-1))

        all_logits.append(output_logits)
        return all_logits, stats
