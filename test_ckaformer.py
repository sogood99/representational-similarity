from ckaformer import CKAFormer
import torch
from torch import nn
import torchvision

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm

import sys


import numpy as np


def test_ckaformer():
    writer = SummaryWriter(log_dir="runs/ckaformer_running_cov")

    fig, ax = plt.subplots()

    n = 500
    d = 784
    classes = 10

    dataset = torchvision.datasets.MNIST(
        root="./dataset",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    X = dataset.data[:n].view(n, -1).float()
    # X = X / X.norm(dim=-1, keepdim=True)
    y = dataset.targets[:n]

    X_test = dataset.data[n : n + 100].view(100, -1).float()
    y_test = dataset.targets[n : n + 100]

    model = CKAFormer(dim=d, depth=500, out_dim=classes)

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
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            out, stats = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            print(loss.item())
            writer.add_scalar("Loss/train", loss.item(), epoch)
            with torch.no_grad():
                acc = (out.argmax(dim=-1) == y).float().mean()
                print("Acc", acc.item())
                writer.add_scalar("Acc/train", acc, epoch)
                writer.add_scalar("Stats/lc", np.mean(stats["lc"]), epoch)
                writer.add_scalar("Stats/rc", np.mean(stats["rc"]), epoch)

            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    out_test, _ = model(X_test)
                    loss_test = criterion(out_test, y_test)
                    writer.add_scalar("Loss/test", loss_test, epoch)
                    acc_test = (out_test.argmax(dim=-1) == y_test).float().mean()
                    writer.add_scalar("Acc/test", acc_test, epoch)


if __name__ == "__main__":
    test_ckaformer()
