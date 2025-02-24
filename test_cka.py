"""
Testing the CKA data on a simple example
"""

import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.animation as animation


from cka import CKA, CKA_derivative


def center(Z):
    return Z - Z.mean(axis=0)


if __name__ == "__main__":
    fig, ax = plt.subplots()

    origin = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    Z = np.array([[-1, 1], [-1, -2], [1, 1], [1, -1], [2, 1]], dtype=np.float32)
    # center the data
    Z = center(Z)
    plt.title("Frame 0: Centering")
    y = np.array([1, 0, 0, 1, 1])
    y_o = y
    # one hot encoding
    y = np.eye(np.max(y) + 1)[y]

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
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
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
            _, (ld, rd) = CKA_derivative(Z, y)

            ld, rd = ld.cpu().numpy(), rd.cpu().numpy()
            Z += ld
        else:
            plt.title(f"Frame {frame}: Right Derivative")

            Z -= rd

        # quiver.set_offsets(Z)
        quiver.set_UVC(Z[:, 0], Z[:, 1])

    # show axes
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    # show the grid
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    ani = animation.FuncAnimation(fig, update, frames=range(21), interval=50)
    plt.show()
