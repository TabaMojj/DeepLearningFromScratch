import numpy as np
from numpy import ndarray


def assert_same_shape(output: ndarray,
                      output_grad: ndarray):
    assert output.shape == output_grad.shape, \
        '''
    Two ndarray should have the same shape; instead, first ndarray's shape is {0}
    and second ndarray's shape is {1}.
    '''.format(tuple(output_grad.shape), tuple(output.shape))
    return None


def assert_dim(t: ndarray,
               dim: int):
    assert len(t.shape) == dim, \
        '''
    Tensor expected to have dimension {0}, instead has dimension {1}
    '''.format(dim, len(t.shape))
    return None


# Padding
def _pad_1d(inp: ndarray,
            num: int) -> ndarray:
    z = np.array([0])
    z = np.repeat(z, num)
    return np.concatenate([z, inp, z])


input_1d = np.array([1, 2, 3, 4, 5])
param_1d = np.array([1, 1, 1])
_pad_1d(input_1d, 1)


# Forward
def conv_1d(inp: ndarray,
            param: ndarray) -> ndarray:
    assert_dim(inp, 1)
    assert_dim(param, 1)

    param_len = param.shape[0]
    param_mid = param_len // 2
    inp_pad = _pad_1d(inp, param_mid)
    out = np.zeros(inp_pad.shape)
    for o in range(out.shape[0]):
        for p in range(param_len):
            out[o] += param[p] * inp_pad[o + p]

    assert_same_shape(inp_pad, out)
    return out


def conv_1d_sum(inp: ndarray,
                param: ndarray) -> ndarray:
    out = conv_1d(inp, param)
    return np.sum(out)


# Gradients
def _param_grad_1d(inp: ndarray,
                   param: ndarray,
                   output_grad: ndarray = None) -> ndarray:
    param_len = param.shape[0]
    param_mid = param_len // 2
    input_pad = _pad_1d(inp, param_mid)
    if output_grad is None:
        output_grad = np.ones_like(input_pad)
    else:
        assert_same_shape(input_pad, output_grad)
    param_grad = np.zeros_like(param)
    # input_grad = np.zeros_like(inp)

    for o in range(inp.shape[0]):
        for p in range(param.shape[0]):
            param_grad[p] += input_pad[o + p] * output_grad[o]
    assert_same_shape(param_grad, param)

    return param_grad


def _input_grad_1d(inp: ndarray,
                   param: ndarray,
                   output_grad: ndarray = None) -> ndarray:
    param_len = param.shape[0]
    param_mid = param_len // 2
    # inp_pad = _pad_1d(inp, param_mid)

    if output_grad is None:
        output_grad = np.ones_like(inp)
    else:
        assert_same_shape(inp, output_grad)

    output_pad = _pad_1d(output_grad, param_mid)

    # param_grad = np.zeros_like(param)
    input_grad = np.zeros_like(inp)

    for o in range(inp.shape[0]):
        for f in range(param.shape[0]):
            input_grad[o] += output_pad[o + param_len - f - 1] * param[f]

    assert_same_shape(input_grad, param)

    return input_grad
