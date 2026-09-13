import numpy as np
import matplotlib.pyplot as plt


def linearFit(x, y):
    A = np.column_stack((np.ones_like(x), x))
    c = np.linalg.lstsq(A, y, rcond=None)[0]

    err = np.linalg.norm(y - A @ c) ** 2
    return c, err


def plotFit(x, y, c):
    x_plot = np.linspace(np.min(x), np.max(x), 100)
    y_plot = c[0] + c[1] * x_plot
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o', x_plot, y_plot, "-")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis("equal")
    plt.show()


if __name__ == "__main__":
    data = np.genfromtxt("data_1_T7.csv", delimiter=",")
    x = np.array(data[:, 0])
    y = np.array(data[:, 1])
    c, err = linearFit(x, y)
    print(f" The linear model is y ~ {c[0]:.2g} + ({c[1]:.2g})x with a least squares error of {err:.4f}")
    plotFit(x, y, c)
