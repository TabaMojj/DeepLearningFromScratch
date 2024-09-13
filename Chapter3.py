from copy import deepcopy

import numpy as np
from numpy import ndarray


def permute_data(X: ndarray, y: ndarray):
    """Permute X and y, using same permutation, along axis=0"""
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


def assert_same_shape(array: ndarray,
                      array_grad: ndarray):
    assert array.shape == array_grad.shape, \
        """
        Two ndarrays should have the same shape;
        instead, first ndarray's shape is {0}
        and second ndarray's shape is {1}.
        """.format(tuple(array_grad.shape), tuple(array.shape))
    return None


class Operation:
    """
    Base class for an operation in a neural network.
    """

    def __init__(self):
        self.input_grad = None
        self.output = None
        self.input_ = None

    def forward(self,
                input_: ndarray,
                inference: bool = False) -> ndarray:
        self.input_ = input_
        self.output = self._output(inference)

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Calls self._intput_grad(). Checks appropriate shapes.
        """
        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad

    def _output(self, inference: bool) -> ndarray:
        """
        The _output method must be defined for each Operation.
        """
        raise NotImplementedError()

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        The _input_grad method must be defined for each Operation.
        """
        raise NotImplementedError()


class ParamOperation(Operation):
    """
    An Operation with parameters.
    """

    def __init__(self, param: ndarray):
        super().__init__()
        self.param_grad = None
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Calls self._input_grad and self._param_grad.
        Checks appropriate shapes.
        """
        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)
        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """
        Every subclass of ParamOperation must implement _param_grad.
        """
        raise NotImplementedError()


class WeightMultiply(ParamOperation):
    """
    Weight multiplication operation for a neural network.
    """

    def __init__(self, W: ndarray):
        """
        Initialize Operation with self.param = W.
        """
        super().__init__(W)

    def _output(self) -> ndarray:
        """
        Compute output.
        """
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Compute input gradient.
        """
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """
        Compute parameter gradient.
        """
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


class BiasAdd(ParamOperation):
    """
    Compute bias addition.
    """

    def __init__(self, B: ndarray):
        """
        Initialize Operation with self.param = B.
        Check appropriate shape.
        """
        assert B.shape[0] == 1
        super().__init__(B)

    def _output(self) -> ndarray:
        """
        Compute output.
        """
        return self.input_ + self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Compute input gradient.
        """
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """
        Compute parameter gradient.
        """
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])


class Dropout(Operation):
    def __init__(self, keep_prob: float = 0.8):
        super().__init__()
        self.keep_prob = keep_prob

    def _output(self, inference: bool) -> ndarray:
        if inference:
            return self.input_ * self.keep_prob
        else:
            self.mask = np.random.binomial(1, self.keep_prob, size=self.input_.shape)
            return self.input_ * self.mask

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad * self.mask


class Sigmoid(Operation):
    """
    Sigmoid activation function.
    """

    def __init__(self):
        super().__init__()

    def _output(self) -> ndarray:
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _intput_grad(self, output_grad: ndarray) -> ndarray:
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad


class Layer:
    """
    A layer of neurons in a neural network.
    """

    def __init__(self, neurons: int):
        """
        The number of neurons roughly corresponds to the breadth of the layer.
        """
        self.output = None
        self.input_ = None
        self.neurons = neurons
        self.first = True
        self.params: list[ndarray] = []
        self.param_grads: list[ndarray] = []
        self.operations: list[Operation] = []

    def _setup_layer(self, num_in: int):
        """
        The setup_layer function must be implemented for each layer.
        """
        raise NotImplementedError()

    def forward(self, input_: ndarray) -> ndarray:
        """
        Passes input forward through a series of operations.
        """
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_
        for operation in self.operations:
            input_ = operation.forward(input_)
        self.output = input_
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Passes output_grad backward through a series of operations.
        Checks appropriate shapes.
        """
        assert_same_shape(self.output, output_grad)
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)
        input_grad = output_grad
        self._param_grads()
        return input_grad

    def _param_grads(self):
        """
        Extracts the _param_grads for a layer's operations.
        """
        # self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self):
        """
        Extracts the _params from a layer's operations.
        """
        # self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)


class Dense(Layer):
    """
    A fully connected layer that inherits from Layer.
    """

    def __init__(self,
                 neurons: int,
                 activation: Operation = Sigmoid(),
                 conv_in: bool = False,
                 dropout: float = 1.0,
                 weight_init: str = "standard"):
        """
        Requires an activation function upon initialization.
        """
        super().__init__(neurons)
        self.activation = activation
        self.conv_in = conv_in
        self.dropout = dropout
        self.weight_init = weight_init

    def _setup_layer(self, input_: ndarray):
        """
        Defines the operations of a fully connected layer.
        """
        # Dynamic attribute. See NeuralNetwork class.
        seed = getattr(self, 'seed', None)
        if seed:
            np.random.seed(seed)
        # self.params = []
        # weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))
        # bias
        self.params.append(np.random.randn(1, self.neurons))
        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]
        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))


class Loss:
    """
    The loss of a neural network.
    """

    def __init__(self):
        self.input_grad = None
        self.target = None
        self.prediction = None

    def forward(self, predictions: ndarray, target: ndarray) -> float:
        """
        Computes the actual loss value.
        """
        assert_same_shape(predictions, target)
        self.prediction = predictions
        self.target = target
        loss_value = self._output()
        return loss_value

    def backward(self) -> ndarray:
        """
        Computes gradients of the loss value with respect to the input to the loss function.
        """
        self.input_grad = self._input_grad()
        assert_same_shape(self.prediction, self.input_grad)
        return self.input_grad

    def _output(self) -> float:
        """
        Every subclass of Loss must implement the _output function.
        """
        raise NotImplementedError()

    def _input_grad(self) -> ndarray:
        """
        Every subclass of Loss must implement the _input_grad function.
        """
        raise NotImplementedError()


class MeanSquaredError(Loss):
    def __init__(self):
        super().__init__()

    def _output(self) -> float:
        """
        Computes the per-observation squared error loss.
        """
        return np.sum(np.power(self.prediction - self.target, 2)) / self.prediction[0]

    def _input_grad(self) -> ndarray:
        """
         Computes the loss gradient with respect to the input for MSE loss.
        """
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]


class NeuralNetwork:
    """
    The class for a neural network.
    """

    def __init__(self, layers: list[Layer],
                 loss: Loss,
                 seed: float = 1):
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward(self, x_batch: ndarray) -> ndarray:
        """
        Passes data forward through a series of layers.
        """
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)
        return x_out

    def backward(self, loss_grad: ndarray):
        """
        Passes data backward through a series of layers.
        """
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_batch(self, X_batch: ndarray, y_batch: ndarray) -> float:
        """
        Passes data forward through the layers.
        Computes the loss.
        Passes daya backward through the layers.
        """
        predictions = self.forward(X_batch)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())
        return loss

    def params(self):
        """
        Gets the parameters for the network.
        """
        for layer in self.layers:
            yield from layer.param_grads

    def param_grads(self):
        """
        Gets the gradient of the loss with respect to the parameters for the network.
        """
        for layer in self.layers:
            yield from layer.param_grads


class Optimizer:
    def __init__(self,
                 lr: float = 0.01,
                 final_lr: float = 0,
                 decay_type: str = None) -> None:
        self.lr = lr
        self.final_lr = final_lr
        self.decay_type = decay_type
        self.first = True

    def _setup_decay(self) -> None:

        if not self.decay_type:
            return
        elif self.decay_type == 'exponential':
            self.decay_per_epoch = np.power(self.final_lr / self.lr,
                                            1.0 / (self.max_epochs - 1))
        elif self.decay_type == 'linear':
            self.decay_per_epoch = (self.lr - self.final_lr) / (self.max_epochs - 1)

    def _decay_lr(self) -> None:

        if not self.decay_type:
            return

        if self.decay_type == 'exponential':
            self.lr *= self.decay_per_epoch

        elif self.decay_type == 'linear':
            self.lr -= self.decay_per_epoch

    def step(self,
             epoch: int = 0) -> None:

        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):
            self._update_rule(param=param, grad=param_grad)

    def _update_rule(self, **kwargs) -> None:
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self,
                 lr: float = 0.01,
                 final_lr: float = 0,
                 decay_type: str = None) -> None:
        super().__init__(lr, final_lr, decay_type)

    def _update_rule(self, **kwargs) -> None:
        update = self.lr * kwargs['grad']
        kwargs['param'] -= update


class SGDMomentum(Optimizer):
    def __init__(self,
                 lr: float = 0.01,
                 final_lr: float = 0,
                 decay_type: str = None,
                 momentum: float = 0.9) -> None:
        super().__init__(lr, final_lr, decay_type)
        self.velocities = None
        self.momentum = momentum

    def step(self) -> None:
        if self.first:
            self.velocities = [np.zeros_like(param)
                               for param in self.net.params()]
            self.first = False

        for (param, param_grad, velocity) in zip(self.net.params(),
                                                 self.net.param_grads(),
                                                 self.velocities):
            self._update_rule(param=param,
                              grad=param_grad,
                              velocity=velocity)

    def _update_rule(self, **kwargs) -> None:

        # Update velocity
        kwargs['velocity'] *= self.momentum
        kwargs['velocity'] += self.lr * kwargs['grad']

        # Use this to update parameters
        kwargs['param'] -= kwargs['velocity']


class Trainer:
    """
    Trains a neural network.
    """

    def __init__(self, net: NeuralNetwork, optim: Optimizer):
        """
        Requires a neural network and an optimizer in order for training to occur.
        Assign the neural network as an instance variable to the optimizer.
        """
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)

    def generate_batches(self,
                         X: ndarray,
                         y: ndarray,
                         size: int = 32) -> tuple[ndarray]:
        assert X.shape[0] == y.shape[0], '''features and target must have the same number of rows,
         instead features has {0} and target has {1}
        '''.format(X.shape[0], y.shape[0])
        N = X.shape[0]
        for i in range(0, N, size):
            X_batch, y_batch = X[i:i + size], y[i:i + size]
            yield X_batch, y_batch

    def fit(self,
            X_train: ndarray,
            y_train: ndarray,
            X_test: ndarray,
            y_test: ndarray,
            epochs: int = 100,
            eval_every: int = 10,
            batch_size: int = 32,
            seed: int = 1,
            restart: bool = True):
        """
        Fits the neural network on the training data for a certain number of epochs.
        Every 'eval_every' epochs, it evaluates the neural network on the testing data.
        """
        np.random.seed(seed)
        if restart:
            self.best_loss = 1e9
            for layer in self.net.layers:
                layer.first = True

        for e in range(epochs):
            if (e + 1) % eval_every == 0:
                # for early stopping
                last_model = deepcopy(self.net)
            X_train, y_train = permute_data(X_train, y_train)
            batch_generator = self.generate_batches(X_train, y_train, batch_size)
            for i, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)
                self.optim.step()
            if (e + 1) % eval_every == 0:
                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, y_test)
                if loss < self.best_loss:
                    print(f"Validation loss after {e + 1} epochs is {loss:.3f}")
                    self.best_loss = loss
                else:
                    print(
                        f"""Loss increased after epoch {e + 1}, final loss was {self.best_loss:.3f}, using the model from epoch {e + 1 - eval_every}""")
                    self.net = last_model
                    # ensure self.optim is still updating self.net
                    setattr(self.optim, 'net', self.net)
                    break
