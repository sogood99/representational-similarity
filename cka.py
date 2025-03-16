import torch
import matplotlib.pyplot as plt


def gram_linear(X):
    """
    Compute Gram matrix for a linear kernel

    :param X: torch.Tensor of shape (n, d)
    """

    return X @ X.t()


def centering_matrix(n):
    """
    Compute the centering matrix C for the given number of points n

    :param n: int, number of points
    """

    H = torch.eye(n) - (1 / n) * torch.ones((n, n))
    return H


def HSIC(X, Y):
    """
    Compute HSIC between X and Y

    :param X: torch.Tensor of shape (n, d)
    :param Y: torch.Tensor of shape (n, d)
    """
    n = X.shape[0]

    K = gram_linear(X)
    L = gram_linear(Y)

    H = centering_matrix(n)

    HSIC = torch.trace(K @ H @ L @ H) / (n - 1) ** 2
    return HSIC


def CKA(X, Y):
    """
    Compute CKA between X and Y

    :param X: torch.Tensor of shape (n, d)
    :param Y: torch.Tensor of shape (n, d)
    """
    HSIC_XY = HSIC(X, Y)
    HSIC_XX = HSIC(X, X)
    HSIC_YY = HSIC(Y, Y)

    CKA = HSIC_XY / (HSIC_XX * HSIC_YY).sqrt()
    return CKA


def CKA_2(X, Y):
    """
    Compute CKA between X and Y, using Frobenius norm

    :param X: torch.Tensor of shape (n, d)
    :param Y: torch.Tensor of shape (n, d)
    """
    G = X.t() @ Y
    K = X.t() @ X
    L = Y.t() @ Y

    frobenius_norm_G = torch.norm(G, p="fro")
    frobenius_norm_K = torch.norm(K, p="fro")
    frobenius_norm_L = torch.norm(L, p="fro")

    return frobenius_norm_G**2 / (frobenius_norm_K * frobenius_norm_L)


def CKA_derivative(X, Y, P=None):
    """
    Compute the derivative of CKA with respect to X, assume it is centered

    :param X: torch.Tensor of shape (n, d)
    :param Y: torch.Tensor of shape (n, d)
    :param P: torch.Tensor of shape (d, d) or None
    """
    if type(X) is not torch.Tensor:
        X = torch.tensor(X, dtype=torch.float32)
    if type(Y) is not torch.Tensor:
        Y = torch.tensor(Y, dtype=X.dtype)
    if type(P) is not torch.Tensor and P is not None:
        P = torch.tensor(P, dtype=X.dtype)

    # Compute the Gram matrices
    K = gram_linear(X)
    L = gram_linear(Y)

    alpha = 2 / (torch.trace(L @ L)).sqrt()
    if P is None:
        ld = alpha * Y @ Y.T @ X / torch.trace(K @ K).sqrt()
    else:
        ld = alpha * P @ P.T @ X / torch.trace(K @ K).sqrt()
    rd = alpha * torch.trace(K @ L) * K @ X / (torch.trace(K @ K) ** 1.5)

    derivative = ld - rd

    return derivative, {
        "ld": ld,
        "rd": rd,
        "lc": (alpha / torch.trace(K @ K).sqrt()).item(),
        "rc": (alpha / (torch.trace(L @ L)).sqrt()).item(),
    }


if __name__ == "__main__":
    # Example usage
    n = 100
    d = 20
    X = torch.randn(n, d)
    X = torch.tensor(X, requires_grad=True)
    Y = torch.randn(n, d)

    X_o = X - X.mean(dim=0)
    Y_o = Y - Y.mean(dim=0)

    print(CKA(X_o, Y_o))
    print(CKA_2(X_o, Y_o))

    derivative, _ = CKA_derivative(X_o, Y_o)

    cka = CKA(X_o, Y_o)
    cka.backward()

    fig, axs = plt.subplots(1, 2)
    axs[0].matshow(derivative.detach().numpy())
    axs[0].set_title("Derivative of CKA")
    axs[1].matshow(X.grad.numpy())
    axs[1].set_title("Gradient of CKA")
    plt.savefig("cka_derivative_verification.png")
    plt.show()
