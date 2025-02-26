"""
Testing the CKA data on a simple example
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.animation as animation

from sklearn.linear_model import LogisticRegression

import torch

from cka import CKA, CKA_derivative


def center(Z):
    return (Z - Z.mean(axis=0)) / np.linalg.norm(Z - Z.mean(axis=0), ord="fro")
    # return Z - Z.mean(axis=0)


if __name__ == "__main__":
    fig, ax = plt.subplots()

    origin = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    Z = np.array([[1, 0], [-1.3, 0], [0, 1.2], [0, -1]], dtype=np.float32)
    # center the data
    Z = center(Z)
    plt.title("Frame 0: Centering")
    y = np.array([1, 1, 0, 0])
    y_o = y
    # one hot encoding
    y = np.eye(np.max(y) + 1)[y]
    # info = 0.1
    # p = np.ones_like(y) / 2 * (1 - info) + y * info

    ## Attn(keys = W_x, queries = X, values = Y.T @ mx)

    lr = LogisticRegression(multi_class="multinomial")

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
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    # ax.set_xlim(-10, 10)
    # ax.set_ylim(-10, 10)

    def update(frame):
        global Z, rd

        if frame % 3 == 0:
            # centering
            Z = center(Z)
            plt.title(f"Frame {frame}: Centering")

        elif frame % 3 == 1:
            plt.title(f"Frame {frame}: Left Derivative")
            lr.fit(Z, y_o)
            p = lr.predict_proba(Z)
            _, (ld, rd) = CKA_derivative(Z, y, p)

            ld, rd = ld.cpu().numpy(), rd.cpu().numpy()
            Z += 0.1 * ld
        else:
            plt.title(f"Frame {frame}: Right Derivative")

            Z -= 0.1 * rd

        # quiver.set_offsets(Z)
        quiver.set_UVC(Z[:, 0], Z[:, 1])

    # show axes
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    # show the grid
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    ani = animation.FuncAnimation(fig, update, frames=range(2000), interval=50)
    plt.show()
