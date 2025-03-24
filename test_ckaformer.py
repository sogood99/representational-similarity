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
    writer = SummaryWriter(log_dir="runs/ckaformer_no_class")

    fig, ax = plt.subplots()

    n = 500
    d = 784
    classes = 10

    train_dataset = torchvision.datasets.MNIST(
        root="./dataset",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./dataset",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=64, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=64, shuffle=False
    )

    model = CKAFormer(dim=d, depth=16, out_dim=classes, num_classes=classes)

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
        global_step = 0
        for epoch in range(10):
            mean_loss = 0
            mean_acc = 0
            for X, y in train_dataloader:
                optimizer.zero_grad()
                out, stats = model(X.view(X.shape[0], -1))
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    acc = (out.argmax(dim=-1) == y).float().mean()
                    mean_loss += loss.item()
                    mean_acc += acc.item()
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("Acc/train", acc.item(), global_step)
                if global_step % 10 == 0:
                    model.eval()
                    mean_loss = 0
                    mean_acc = 0
                    for X, y in test_dataloader:
                        out, _ = model(X.view(X.shape[0], -1))
                        loss = criterion(out, y)
                        acc = (out.argmax(dim=-1) == y).float().mean()
                        mean_loss += loss.item()
                        mean_acc += acc.item()
                    mean_loss /= len(test_dataloader)
                    mean_acc /= len(test_dataloader)
                    writer.add_scalar("Loss/test", mean_loss, global_step)
                    writer.add_scalar("Acc/test", mean_acc, global_step)
                global_step += 1
            mean_loss /= len(train_dataloader)
            mean_acc /= len(train_dataloader)


if __name__ == "__main__":
    test_ckaformer()
