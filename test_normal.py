from ckaformer import LinearClassifier
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


def test_normal():
    writer = SummaryWriter(log_dir="runs/linear_classifier")

    dataset = datasets.MNIST(
        root="./dataset",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    n = 500
    d = 784
    classes = 10

    model = LinearClassifier(d, classes)

    X = dataset.data[:n].view(n, -1).float()
    y = dataset.targets[:n]

    X_test = dataset.data[n : n + 100].view(100, -1).float()
    y_test = dataset.targets[n : n + 100]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for i in range(100):
        optimizer.zero_grad()
        P = model(X)
        loss = criterion(P, y)
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", loss, i)

        with torch.no_grad():
            acc = (P.argmax(dim=-1) == y).float().mean()
            writer.add_scalar("Acc/train", acc, i)

        if i % 10 == 0:
            with torch.no_grad():
                P_test = model(X_test)
                loss_test = criterion(P_test, y_test)
                acc_test = (P_test.argmax(dim=-1) == y_test).float().mean()
                writer.add_scalar("Loss/test", loss_test, i)
                writer.add_scalar("Acc/test", acc_test, i)


if __name__ == "__main__":
    test_normal()
