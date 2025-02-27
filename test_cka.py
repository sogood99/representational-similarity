"""
Testing the CKA data on a simple example
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.animation as animation

from sklearn.linear_model import LogisticRegression

import torch

from cka import CKA_derivative


def center(Z):
    return (Z - Z.mean(axis=0)) / np.linalg.norm(Z - Z.mean(axis=0), ord="fro")
    # return Z - Z.mean(axis=0)


if __name__ == "__main__":
    fig, ax = plt.subplots()

    n = 10
    classes = 5
    d = 2

    # origin = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    np.random.seed(1234)
    origin = np.zeros((2, n))
    # Z = np.array([[1, 0], [-1.3, 0], [0, 1.2], [0, -1]], dtype=np.float32)
    Z = np.random.randn(n, d)
    # center the data
    Z = center(Z)
    plt.title("Frame 0: Centering")
    # y = np.array([1, 1, 0, 0])
    y = np.random.randint(0, classes, n)
    y_o = y
    # one hot encoding
    y = np.eye(np.max(y) + 1)[y]
    print(y.shape)

    ## Attn(keys = W_x, queries = X, values = Y.T @ mx)

    # clf = LogisticRegression(multi_class="multinomial")
    clf = torch.nn.Sequential(
        torch.nn.Linear(d, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, classes),
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)

    colors = matplotlib.cm.get_cmap("tab10")

    quiver = ax.quiver(
        *origin,
        Z[:, 0],
        Z[:, 1],
        color=colors(y_o),
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    def update(frame):
        global Z, rd

        lr = max(0.2 - frame * 0.001, 0.0001)

        if frame % 3 == 0:
            # centering
            Z = center(Z)
            plt.title(f"Frame {frame}: Centering")

        elif frame % 3 == 1:
            plt.title(f"Frame {frame}: Left Derivative")
            # clf.fit(Z, y_o)
            # p = clf.predict_proba(Z)
            p = clf(torch.tensor(Z, dtype=torch.float32)).detach().numpy()
            _, (ld, rd) = CKA_derivative(Z, y, p)

            ld, rd = ld.cpu().numpy(), rd.cpu().numpy()
            Z += lr * ld
        else:
            plt.title(f"Frame {frame}: Right Derivative")

            Z -= lr * rd

        # quiver.set_offsets(Z)
        quiver.set_UVC(Z[:, 0], Z[:, 1])

    # show axes
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    # show the grid
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    ani = animation.FuncAnimation(fig, update, frames=1000, interval=50)
    plt.show()
