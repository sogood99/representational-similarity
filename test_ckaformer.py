from ckaformer import CKAFormer
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm

import sys


import numpy as np


def test_ckaformer():
    fig, ax = plt.subplots()

    n = 100
    d = 784
    classes = 10

    # X = torch.randn(n, d)
    # X = X / X.norm(dim=-1, keepdim=True)
    dataset = torchvision.datasets.MNIST(
        root="./dataset",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    X = dataset.data[:n].view(n, -1).float()
    X = X / X.norm(dim=-1, keepdim=True)
    y = dataset.targets[:n]

    model = CKAFormer(dim=d, depth=500, out_dim=classes, y=y)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    if len(sys.argv) > 1:
        losses = []

        origin = np.zeros((d, n))

        colors = plt.cm.get_cmap("tab10")

        quiver = ax.quiver(
            *origin,
            X[:, 0],
            X[:, 1],
            color=colors(y.cpu().numpy()),
            angles="xy",
            scale_units="xy",
            scale=1,
        )
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        def update(frame):
            optimizer.zero_grad()
            new_X, out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            new_X = new_X.detach().cpu().numpy()

            quiver.set_UVC(new_X[:, 0], new_X[:, 1])
            print(loss.item())

            with torch.no_grad():
                print("Acc", (out.argmax(dim=-1) == y).float().mean().item())

        ani = animation.FuncAnimation(fig, update, frames=100, interval=100)
        plt.show()
    else:
        for _ in range(100):
            optimizer.zero_grad()
            new_X, out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            print(loss.item())
            with torch.no_grad():
                print("Acc", (out.argmax(dim=-1) == y).float().mean().item())


if __name__ == "__main__":
    test_ckaformer()
