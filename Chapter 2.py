from typing import Dict, Tuple

import numpy as np
from numpy import ndarray


def forward_linear_regression(X_batch: ndarray,
                              y_batch: ndarray,
                              weights: Dict[str, ndarray]) -> Tuple[float, Dict[str, ndarray]]:
    """forward pass for the step-by-step linear regression."""

    # assert batch sizes of X and y are equal
    assert X_batch.shape[0] == y_batch.shape[0]

    # assert matrix multiplication is applicable
    assert X_batch.shape[1] == weights['W'].shape[0]

    # assert B is 1x1 ndarray
    assert weights['B'].shape[0] == weights['B'].shape[1] == 1

    # compute the operations on the forward pass
    N = np.dot(X_batch, weights['W'])
    P = N + weights['B']
    loss = np.mean(np.power(y_batch - P, 2))

    # save the information computed on the forward pass
    forward_info: Dict[str, ndarray] = {'X': X_batch, 'N': N, 'P': P, 'y': y_batch}

    return loss, forward_info


def loss_gradients(forward_info: Dict[str, ndarray],
                   weights: Dict[str, ndarray]) -> Dict[str, ndarray]:
    """Compute dLdW and dLdB for the step-by-step linear regression model."""

    batch_size = forward_info['X'].shape[0]
    dLdP = -2 * (forward_info['y'] - forward_info['P'])
    dPdN = np.ones_like(forward_info['N'])
    dPdB = np.ones_like(weights['B'])
    dNdW = np.transpose(forward_info['X'], (1, 0))
    dLdN = dLdP * dPdN
    dLdW = np.dot(dNdW, dLdN)
    dLdB = (dLdP * dPdB).sum(axis=0)

    return {'W': dLdW, 'B': dLdB}
