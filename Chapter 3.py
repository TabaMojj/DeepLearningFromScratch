import numpy as np
from numpy import ndarray


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

    def forward(self, input_: ndarray):
        """
        Stores input in the self._input instance variable
        """
        self.input_ = input_
        self.output = self._output()
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Calls self._intput_grad(). Checks appropriate shapes.
        """
        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad

    def _output(self) -> ndarray:
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

    def __init__(self, neurons: int, activation: Operation = Sigmoid()):
        """
        Requires an activation function upon initialization.
        """
        super().__init__(neurons)
        self.activation = activation

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


