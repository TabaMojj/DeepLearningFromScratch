import numpy as np
from numpy import ndarray

Batch = tuple[ndarray, ndarray]


def forward_linear_regression(X_batch: ndarray,
                              y_batch: ndarray,
                              weights: dict[str, ndarray]) -> tuple[dict[str, ndarray], float]:
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
    forward_info: dict[str, ndarray] = {'X': X_batch, 'N': N, 'P': P, 'y': y_batch}

    return forward_info, loss


def loss_gradients_regression(forward_info: dict[str, ndarray],
                              weights: dict[str, ndarray]) -> dict[str, ndarray]:
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


def to_2d_np(a: ndarray, type: str = 'col') -> ndarray:
    """Turns a 1D Tensor into 2D"""
    assert a.ndim == 1, "Input tensors must be 1 dimensional"
    if type == 'col':
        return a.reshape(-1, 1)
    elif type == 'row':
        return a.reshape(1, -1)


def permute_data(X: ndarray, y: ndarray):
    """Permute X and y, using same permutation, along axis=0"""
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


def generate_batch(X: ndarray,
                   y: ndarray,
                   start: int = 0,
                   batch_size: int = 10) -> Batch:
    """Given a start position, generate batch from X and y."""
    assert X.ndim == y.ndim == 2, "X and y must be 2 dimensional"

    if start + batch_size > X.shape[0]:
        batch_size = X.shape[0] - start

    X_batch, y_batch = X[start:start + batch_size], y[start:start + batch_size]
    return X_batch, y_batch


def init_weights(n_in: int) -> dict[str, ndarray]:
    """Initialize weights on first forward pass of model."""
    return {'W': np.random.randn(n_in, 1), 'B': np.random.randn(1, 1)}


def train(X: ndarray,
          y: ndarray,
          n_iter: int = 1000,
          learning_rate: float = 0.01,
          batch_size: int = 100,
          return_losses: bool = False,
          return_weights: bool = False,
          seed: int = 1) -> None | tuple[list[float], dict[str, ndarray]]:
    """
    Train model for a certain number of epochs.
    """
    if seed:
        np.random.seed(seed)
    start = 0
    weights = init_weights(X.shape[1])
    X, y = permute_data(X, y)
    if return_losses:
        losses = []
    for i in range(n_iter):
        if start >= X.shape[0]:
            X, y = permute_data(X, y)
            start = 0
        X_batch, y_batch = generate_batch(X, y, start, batch_size)
        start += batch_size
        forward_info, loss = forward_linear_regression(X_batch, y_batch, weights)
        if return_losses:
            losses.append(loss)
        loss_grads = loss_gradients_regression(forward_info, weights)
        for key in weights.keys():
            weights[key] -= learning_rate * loss_grads[key]
    if return_weights:
        return losses, weights
    return None


def predict(X: ndarray, weights: dict[str, ndarray]):
    """Generate predictions from the step-by-step linear regression model."""
    N = np.dot(X, weights['W'])
    return N + weights['B']


def mae(preds: ndarray, actual: ndarray):
    """Compute mean absolute error."""
    return np.mean(np.abs(preds - actual))


def rmse(preds: ndarray, actual: ndarray):
    """Compute root mean squared error."""
    return np.sqrt(np.mean(np.power(preds - actual, 2)))


def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))


def forward_loss(X: ndarray,
                 y: ndarray,
                 weights: dict[str, ndarray]
                 ) -> tuple[dict[str, ndarray], float]:
    """Compute the forward pass and the loss for the step-by-step neural network model."""
    M1 = np.dot(X, weights['W1'])
    N1 = M1 + weights['B1']
    O1 = sigmoid(N1)
    M2 = np.dot(O1, weights['W2'])
    P = M2 + weights['B2']
    loss = np.mean(np.power(y - P, 2))
    forward_info = {'X': X, 'M1': M1, 'N1': N1, 'O1': O1, 'M2': M2, 'P': P, 'y': y}
    return forward_info, loss


def loss_gradients(forward_info: dict[str, ndarray],
                   weights: dict[str, ndarray]) -> dict[str, ndarray]:
    """
    Compute the partial derivatives of the loss with respect to each of the parameters in the neural network.
    """
    dLdP = -(forward_info['y'] - forward_info['P'])
    dPdM2 = np.ones_like(forward_info['M2'])
    dLdM2 = dLdP * dPdM2
    dPdB2 = np.ones_like(weights['B2'])
    dLdB2 = (dLdP * dPdB2).sum(axis=0)
    dM2dW2 = np.transpose(forward_info['O1'], (1, 0))
    dLdW2 = np.dot(dM2dW2, dLdP)
    dM2dO1 = np.transpose(weights['W2'], (1, 0))
    dLdO1 = np.dot(dLdM2, dM2dO1)
    dO1dN1 = sigmoid(forward_info['N1']) * (1 - sigmoid(forward_info['N1']))
    dLdN1 = dLdO1 * dO1dN1
    dN1dB1 = np.ones_like(weights['B1'])
    dN1dM1 = np.ones_like(forward_info['M1'])
    dLdB1 = (dLdN1 * dN1dB1).sum(axis=0)
    dLdM1 = dLdN1 * dN1dM1
    dM1dW1 = np.transpose(forward_info['X'], (1, 0))
    dLdW1 = np.dot(dM1dW1, dLdM1)
    return {'W2': dLdW2, 'B2': dLdB2.sum(axis=0), 'W1': dLdW1, 'B1': dLdB1.sum(axis=0)}


def init_weights_nn(input_size: int,
                    hidden_size: int) -> dict[str, ndarray]:
    """
    Initialize weights during the forward pass for step-by-step neural network model.
    """
    weights = {'W1': np.random.randn(input_size, hidden_size),
               'B1': np.random.randn(1, hidden_size), 'W2': np.random.randn(hidden_size, 1),
               'B2': np.random.randn(1, 1)}
    return weights


def train_nn(X_train: ndarray, y_train: ndarray,
             X_test: ndarray, y_test: ndarray,
             n_iter: int = 1000,
             test_every: int = 1000,
             learning_rate: float = 0.01,
             hidden_size=13,
             batch_size: int = 100,
             return_losses: bool = False,
             return_weights: bool = False,
             return_scores: bool = False,
             seed: int = 1):
    if seed:
        np.random.seed(seed)

    start = 0

    # Initialize weights
    weights = init_weights_nn(X_train.shape[1],
                           hidden_size=hidden_size)

    # Permute data
    X_train, y_train = permute_data(X_train, y_train)

    losses = []

    val_scores = []

    for i in range(n_iter):

        # Generate batch
        if start >= X_train.shape[0]:
            X_train, y_train = permute_data(X_train, y_train)
            start = 0

        X_batch, y_batch = generate_batch(X_train, y_train, start, batch_size)
        start += batch_size

        # Train net using generated batch
        forward_info, loss = forward_loss(X_batch, y_batch, weights)

        if return_losses:
            losses.append(loss)

        loss_grads = loss_gradients(forward_info, weights)
        for key in weights.keys():
            weights[key] -= learning_rate * loss_grads[key]

        if return_scores:
            if i % test_every == 0 and i != 0:
                preds = predict(X_test, weights)
                val_scores.append(mae(preds, y_test))

    if return_weights:
        return losses, weights, val_scores

    return None
