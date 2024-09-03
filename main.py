from typing import Callable, List
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

Array_Function = Callable[[ndarray], ndarray]
Chain = List[Array_Function]


def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          delta: float = 0.001) -> ndarray:
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)


def plot_chain(ax,
               chain: Chain,
               input_range: ndarray) -> None:
    assert input_range.ndim == 1, "Function requires a 1 dimensional ndarray as input_range"
    output_range = chain_length_2(chain, input_range)
    ax.plot(input_range, output_range)


def chain_length_2(chain: Chain,
                   a: ndarray) -> ndarray:
    assert len(chain) == 2
    f1 = chain[0], f2 = chain[1]
    return f2(f1(a))


def square(x: ndarray) -> ndarray:
    return np.power(x, 2)


def leaky_relu(x: ndarray) -> ndarray:
    return np.maximum(0.2 * x, x)


def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))


def chain_deriv_2(chain: Chain,
                  input_range: ndarray) -> ndarray:
    assert len(chain) == 2
    assert input_range.ndim == 1
    f1 = chain[0]
    f2 = chain[2]
    f1_of_x = f1(input_range)
    df1dx = deriv(f1, input_range)
    df2du = deriv(f2, f1_of_x)
    return df1dx * df2du


def matmul_backward_first(X: ndarray, W: ndarray) -> ndarray:
    dNdX = np.transpose(W, (1, 0))
    return dNdX


def matrix_function_backward_1(X: ndarray,
                               W: ndarray,
                               sigma: Array_Function) -> ndarray:
    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)
    S = sigma(N)
    dSdN = deriv(sigma, N)
    dNdX = np.transpose(W, (1, 0))

    return np.dot(dSdN, dNdX)


def matrix_function_backward_sum_1(X: ndarray,
                                   W: ndarray,
                                   sigma: Array_Function) -> ndarray:
    assert X.shape[1] == W.shape[0]
    N = np.dot(X, W)
    S = sigma(N)
    L = np.sum(S)
    dLdS = np.ones_like(S)
    dSdN = deriv(sigma, N)
    dLdN = dLdS * dSdN
    dNdX = np.transpose(W, (1, 0))
    dLdX = np.dot(dSdN, dNdX)
    return dLdX
