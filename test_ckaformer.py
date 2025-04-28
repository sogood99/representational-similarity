import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from ckaformer import CKAFormer


def test_ckaformer(
        d: int = 784,
        classes: int = 10,
        gamma: float = 1e-4,
        depth: int = 32,
        trainable_mean: bool = True,
):
    writer = SummaryWriter(
        log_dir="debug-runs/ckaformer_{}_{}_{}".format(
            gamma, depth, "trainable" if trainable_mean else "fixed"
        )
    )

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
        dataset=train_dataset, batch_size=250, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=250, shuffle=False
    )

    model = CKAFormer(
        dim=d,
        depth=depth,
        out_dim=classes,
        num_classes=classes,
        trainable_mean=trainable_mean,
        gamma=gamma,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train()

    global_step = 0
    for epoch in trange(10, desc="Epochs"):
        mean_loss = 0
        mean_acc = 0
        for X, y in train_dataloader:
            optimizer.zero_grad()
            all_logits, stats = model(X.flatten(start_dim=1))
            layerwise_loss = {
                "Loss/train/layer{}".format(i): criterion(logits, y)
                for i, logits in enumerate(all_logits[:-1])
            }
            last_layer_out = all_logits[-1]
            layerwise_loss["Loss/train"] = criterion(last_layer_out, y)

            sum(layerwise_loss.values()).backward()
            optimizer.step()

            with torch.no_grad():
                acc = (last_layer_out.argmax(dim=-1) == y).float().mean()
                mean_loss += layerwise_loss["Loss/train"].item()
                mean_acc += acc.item()
            for key, val in layerwise_loss.items():
                writer.add_scalar(key, val, global_step)
            writer.add_scalar("Acc/train", acc.item(), global_step)
            if global_step % 10 == 0:
                model.eval()
                mean_loss = 0
                mean_acc = 0
                for X, y in test_dataloader:
                    all_logits, _ = model(X.view(X.shape[0], -1))
                    out = all_logits[-1]
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
    import jsonargparse

    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(test_ckaformer)
    args = parser.parse_args()

    test_ckaformer(**args.as_dict())
