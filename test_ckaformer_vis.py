import os
import sys
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ckaformer import CKAFormer
from repsim import AngularCKA
from repsim.stats import ManifoldPCA

from sklearn.manifold import MDS
from sklearn.decomposition import PCA

# === PCA Visualization ===


def low_d_vis_1(hiddens: dict[str, torch.Tensor], metric, n_dim_mds=5, n_dim_pca=3):
    # First, embed all matrices of neural activity as 'points' in the metric space
    points = {k: metric.neural_data_to_point(x) for k, x in hiddens.items()}

    print(points.keys())

    # Second, compute pairwise distances
    pair_dist = torch.zeros(len(hiddens), len(hiddens))
    for i, p_i in enumerate(points.values()):
        for j, p_j in enumerate(points.values()):
            # The 'length' computation happens on-device (cuda), so we'll grab
            # the scalar result back to CPU with 'Tensor.item()'
            pair_dist[i, j] = metric.length(p_i, p_j).item()

    # Third, use MDS to find a Euclidean embedding of these pairwise distances
    mds = MDS(n_components=n_dim_mds, metric=True, dissimilarity="precomputed")
    mds_embedding = mds.fit_transform(pair_dist)

    # Fourth, use PCA to further reduce
    pcs = PCA(n_components=n_dim_pca).fit_transform(mds_embedding)

    # Determine indices into the rows of 'pcs' to plot
    idx_targets = list(hiddens.keys()).index("labels") if "labels" in hiddens else None
    idx_input = list(hiddens.keys()).index("inputs") if "inputs" in hiddens else None
    path_indices = [i for i in range(len(hiddens)) if i not in (idx_input, idx_targets)]
    if idx_input is not None:
        path_indices = [idx_input] + path_indices

    # Plot grid
    vmin, vmax = pcs.min(), pcs.max()
    vmin, vmax = (vmin + vmax) / 2 - (vmax - vmin) * 5 / 8, (vmin + vmax) / 2 + (
        vmax - vmin
    ) * 5 / 8
    fig, ax = plt.subplots(
        n_dim_pca, n_dim_pca, figsize=(n_dim_pca * 2.5, n_dim_pca * 2.5)
    )
    for i, pc_i in enumerate(pcs.T):
        for j, pc_j in enumerate(pcs.T):
            if j >= i:
                ax[i, j].remove()
                continue

            ax[i, j].plot(pc_j[path_indices], pc_i[path_indices], marker=".")
            if idx_input is not None:
                ax[i, j].plot(pc_j[idx_input], pc_i[idx_input], color="k", marker="s")
            if idx_targets is not None:
                ax[i, j].plot(
                    pc_j[idx_targets], pc_i[idx_targets], color="m", marker="*"
                )
            ax[i, j].set_xlim([vmin, vmax])
            ax[i, j].set_ylim([vmin, vmax])
            ax[i, j].set_xlabel(f"PC {j+1}")
            ax[i, j].set_ylabel(f"PC {i+1}")
            ax[i, j].grid()
    fig.tight_layout()
    return fig


def low_d_vis_2(hiddens: dict[str, torch.Tensor], metric, n_dim_pca=3):
    # Embed matrices of neural activity as 'points' in the metric space
    points = {k: metric.neural_data_to_point(x) for k, x in hiddens.items()}

    # PCA on the manifold
    mpca = ManifoldPCA(metric, n_components=n_dim_pca)
    pcs = mpca.fit_transform(points.values()).cpu()

    # Determine indices for plotting
    idx_targets = list(hiddens.keys()).index("labels") if "labels" in hiddens else None
    idx_input = list(hiddens.keys()).index("inputs") if "inputs" in hiddens else None
    path_indices = [i for i in range(len(hiddens)) if i not in (idx_input, idx_targets)]
    if idx_input is not None:
        path_indices = [idx_input] + path_indices

    # Plotting
    vmin, vmax = -torch.pi / 2, torch.pi / 2
    fig, ax = plt.subplots(
        n_dim_pca, n_dim_pca, figsize=(n_dim_pca * 2.5, n_dim_pca * 2.5)
    )
    for i, pc_i in enumerate(pcs.T):
        for j, pc_j in enumerate(pcs.T):
            if j >= i:
                ax[i, j].remove()
                continue
            ax[i, j].plot(pc_j[path_indices], pc_i[path_indices], marker=".")
            if idx_input is not None:
                ax[i, j].plot(pc_j[idx_input], pc_i[idx_input], color="k", marker="s")
            if idx_targets is not None:
                ax[i, j].plot(
                    pc_j[idx_targets], pc_i[idx_targets], color="m", marker="*"
                )
            ax[i, j].set_xlim([vmin, vmax])
            ax[i, j].set_ylim([vmin, vmax])
            ax[i, j].set_xlabel(f"PC {j+1} (radians)")
            ax[i, j].set_ylabel(f"PC {i+1} (radians)")
    fig.tight_layout()
    return fig


# === Main Training and Visualization ===


def test_ckaformer():
    writer = SummaryWriter(log_dir="runs/ckaformer_vis")
    os.makedirs("pca_vis", exist_ok=True)

    d = 784
    classes = 10

    # Load MNIST
    transform = torchvision.transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(
        root="./dataset", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./dataset", train=False, download=True, transform=transform
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )

    # Model, loss, optimizer
    model = CKAFormer(
        dim=d,
        depth=64,
        out_dim=classes,
        num_classes=classes,
        gamma=1e-5,
        save_hidden=True,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    if len(sys.argv) > 1:
        print("Skipping visualization mode (interactive animation not yet updated).")
        return

    # Training Loop
    global_step = 0
    for epoch in range(10):
        mean_loss = 0
        mean_acc = 0
        for X, y in train_dataloader:
            X = X.view(X.shape[0], -1)
            optimizer.zero_grad()
            out, stats = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = (out.argmax(dim=-1) == y).float().mean()
                mean_loss += loss.item()
                mean_acc += acc.item()
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar("Acc/train", acc.item(), global_step)

            # Evaluate on test set
            if global_step % 10 == 0:
                model.eval()
                mean_loss_test = 0
                mean_acc_test = 0
                for X_test, y_test in test_dataloader:
                    X_test = X_test.view(X_test.shape[0], -1)
                    out_test, _ = model(X_test)
                    loss_test = criterion(out_test, y_test)
                    acc_test = (out_test.argmax(dim=-1) == y_test).float().mean()
                    mean_loss_test += loss_test.item()
                    mean_acc_test += acc_test.item()
                mean_loss_test /= len(test_dataloader)
                mean_acc_test /= len(test_dataloader)
                writer.add_scalar("Loss/test", mean_loss_test, global_step)
                writer.add_scalar("Acc/test", mean_acc_test, global_step)
                model.train()

            # PCA Visualization
            if global_step % 50 == 0:
                model.eval()
                with torch.no_grad():
                    import torch.nn.functional as F

                    X_vis, y_vis = next(iter(test_dataloader))
                    print(y_vis)
                    y_vis = F.one_hot(y_vis, num_classes=classes)
                    X_vis_flat = X_vis.view(X_vis.shape[0], -1)
                    out_vis, stats_vis = model(X_vis_flat)
                    print(stats_vis["hidden"][0].shape)

                    # Convert list of hidden activations to dict
                    hiddens = {}
                    hiddens["inputs"] = X_vis_flat.float()
                    for i, t in enumerate(stats_vis["hidden"]):
                        hiddens[f"layer{i}"] = t.float()
                    hiddens["labels"] = y_vis.float()

                    # Use the first hidden layer to initialize the metric
                    metric = AngularCKA(m=64)

                    fig = low_d_vis_1(hiddens, metric, n_dim_pca=3)
                    fig.savefig(f"pca_vis/step_{global_step}.png")
                    plt.close(fig)
                model.train()

            global_step += 1

        mean_loss /= len(train_dataloader)
        mean_acc /= len(train_dataloader)


if __name__ == "__main__":
    test_ckaformer()
