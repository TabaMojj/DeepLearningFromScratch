from typing import Union

import numpy as np
from numpy import ndarray
from scipy.special import logsumexp

from Chapter3 import assert_same_shape

Numerable = Union[float, int]


def ensure_number(num: Numerable) -> 'NumberWithGrad':
    if isinstance(num, NumberWithGrad):
        return num
    else:
        return NumberWithGrad(num)


def sigmoid(x: ndarray):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x: ndarray):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x: ndarray):
    return np.tanh(x)


def dtanh(x: ndarray):
    return 1 - np.tanh(x) * np.tanh(x)


def softmax(x, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def batch_softmax(input_array: ndarray):
    out = []
    for row in input_array:
        out.append(softmax(row, axis=1))
    return np.stack(out)


class NumberWithGrad:
    def __init__(self,
                 num: Numerable,
                 depends_on: list['NumberWithGrad'] = None,
                 creation_op: str = ''):
        self.num = num
        self.grad = None
        self.depends_on = depends_on or []
        self.creation_op = creation_op

    def __add__(self,
                other: Numerable) -> 'NumberWithGrad':
        return NumberWithGrad(self.num + ensure_number(other).num,
                              depends_on=[self, ensure_number(other)],
                              creation_op='add')

    def __mul__(self,
                other: Numerable = None) -> 'NumberWithGrad':
        return NumberWithGrad(self.num * ensure_number(other).num,
                              depends_on=[self, ensure_number(other)],
                              creation_op='mul')

    def backward(self,
                 backward_grad: Numerable = None) -> None:
        if backward_grad is None:  # first time calling backward
            self.grad = 1
        else:
            # These lines allow gradients to accumulate.
            # If the gradient doesn't exist yet, simply set it equal
            # to backward_grad
            if self.grad is None:
                self.grad = backward_grad
            else:
                self.grad += backward_grad

        if self.creation_op == 'add':
            # Simply send backward self.grad,
            # since increasing either of these elements will increase the output by that same amount.
            self.depends_on[0].backward(self.grad)
            self.depends_on[1].backward(self.grad)
        if self.creation_op == 'mul':
            # Calculate the derivative with  respect to the first element.
            new = self.depends_on[1] * self.grad
            # Send backward the derivative with respect to that element
            self.depends_on[0].backward(new.num)

            # Calculate the derivative with respect to the second element
            new = self.depends_on[0] * self.grad
            # Send backward the derivative with respect to that element
            self.depends_on[1].backward(new.num)


class RNNNode:

    def __init__(self):
        self.X_out = None
        self.H_out = None
        self.H_int = None
        self.Z = None
        self.H_in = None
        self.X_in = None

    def forward(self,
                x_in: ndarray,
                H_in: ndarray,
                params_dict: dict[str, dict[str, ndarray]]
                ) -> tuple[ndarray]:
        self.X_in = x_in
        self.H_in = H_in
        self.Z = np.column_stack((x_in, H_in))
        self.H_int = np.dot(self.Z, params_dict['W_f']['value']) + params_dict['B_f']['value']
        self.H_out = tanh(self.H_int)
        self.X_out = np.dot(self.H_out, params_dict['W_v']['value']) + params_dict['B_v']['value']
        return self.X_out, self.H_out

    def backward(self,
                 X_out_grad: ndarray,
                 H_out_grad: ndarray,
                 params_dict: dict[str, dict[str, ndarray]]) -> tuple[ndarray]:
        assert_same_shape(X_out_grad, self.X_out)
        assert_same_shape(H_out_grad, self.H_out)

        params_dict['B_v']['deriv'] += X_out_grad.sum(axis=0)
        params_dict['W_v']['deriv'] += np.dot(self.H_out.T, X_out_grad)

        dh = np.dot(X_out_grad, params_dict['W_v']['value'].T)
        dh += H_out_grad

        dH_int = dh * dtanh(self.H_int)

        params_dict['B_f']['deriv'] += dH_int.sum(axis=0)
        params_dict['W_f']['deriv'] += np.dot(self.Z.T, dH_int)

        dz = np.dot(dH_int, params_dict['W_f']['value'].T)

        X_in_grad = dz[:, :self.X_in.shape[1]]
        H_in_grad = dz[:, self.X_in.shape[1]:]

        assert_same_shape(X_out_grad, self.X_out)
        assert_same_shape(H_out_grad, self.H_out)

        return X_in_grad, H_in_grad


class GRUNode:

    def __init__(self):
        self.X_out = None
        self.H_out = None
        self.H_h = None
        self.h_bar = None
        self.h_bar_int = None
        self.X_h = None
        self.r = None
        self.u = None
        self.h_reset = None
        self.u_int = None
        self.r_int = None
        self.H_u = None
        self.X_u = None
        self.X_r = None
        self.H_r = None
        self.H_in = None
        self.X_in = None

    def forward(self,
                X_in: ndarray,
                H_in: ndarray,
                params_dict: dict[str, dict[str, ndarray]]) -> tuple[ndarray]:
        self.X_in = X_in
        self.H_in = H_in

        # reset gate
        self.X_r = np.dot(X_in, params_dict['W_xr']['value'])
        self.H_r = np.dot(H_in, params_dict['W_hr']['value'])

        # update gate        
        self.X_u = np.dot(X_in, params_dict['W_xu']['value'])
        self.H_u = np.dot(H_in, params_dict['W_hu']['value'])

        # gates   
        self.r_int = self.X_r + self.H_r + params_dict['B_r']['value']
        self.r = sigmoid(self.r_int)

        self.u_int = self.X_r + self.H_r + params_dict['B_u']['value']
        self.u = sigmoid(self.u_int)

        # new state        
        self.h_reset = self.r * H_in
        self.X_h = np.dot(X_in, params_dict['W_xh']['value'])
        self.H_h = np.dot(self.h_reset, params_dict['W_hh']['value'])
        self.h_bar_int = self.X_h + self.H_h + params_dict['B_h']['value']
        self.h_bar = tanh(self.h_bar_int)

        self.H_out = self.u * self.H_in + (1 - self.u) * self.h_bar
        self.X_out = np.dot(self.H_out, params_dict['W_v']['value']) + params_dict['B_v']['value']

        return self.X_out, self.H_out

    def backward(self,
                 X_out_grad: ndarray,
                 H_out_grad: ndarray,
                 params_dict: dict[str, dict[str, ndarray]]):
        params_dict['B_v']['deriv'] += X_out_grad.sum(axis=0)
        params_dict['W_v']['deriv'] += np.dot(self.H_out.T, X_out_grad)

        dh_out = np.dot(X_out_grad, params_dict['W_v']['value'].T)
        dh_out += H_out_grad

        du = self.H_in * H_out_grad - self.h_bar * H_out_grad
        dh_bar = (1 - self.u) * H_out_grad

        dh_bar_int = dh_bar * dtanh(self.h_bar_int)
        params_dict['B_h']['deriv'] += dh_bar_int.sum(axis=0)
        params_dict['W_xh']['deriv'] += np.dot(self.X_in.T, dh_bar_int)

        dX_in = np.dot(dh_bar_int, params_dict['W_xh']['value'].T)

        params_dict['W_hh']['deriv'] += np.dot(self.h_reset.T, dh_bar_int)
        dh_reset = np.dot(dh_bar_int, params_dict['W_hh']['value'].T)

        dr = dh_reset * self.H_in
        dH_in = dh_reset * self.r

        # update branch
        du_int = dsigmoid(self.u_int) * du
        params_dict['B_u']['deriv'] += du_int.sum(axis=0)

        dX_in += np.dot(du_int, params_dict['W_xu']['value'].T)
        params_dict['W_xu']['deriv'] += np.dot(self.X_in.T, du_int)

        dH_in += np.dot(du_int, params_dict['W_hu']['value'].T)
        params_dict['W_hu']['deriv'] += np.dot(self.H_in.T, du_int)

        # reset branch
        dr_int = dsigmoid(self.r_int) * dr
        params_dict['B_r']['deriv'] += dr_int.sum(axis=0)

        dX_in += np.dot(dr_int, params_dict['W_xr']['value'].T)
        params_dict['W_xr']['deriv'] += np.dot(self.X_in.T, dr_int)

        dH_in += np.dot(dr_int, params_dict['W_hr']['value'].T)
        params_dict['W_hr']['deriv'] += np.dot(self.H_in.T, dr_int)

        return dX_in, dH_in


class RNNLayer:
    def __init__(self,
                 hidden_size: int,
                 output_size: int,
                 weight_scale: float = None):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight_scale = weight_scale
        self.start_H = np.zeros((1, hidden_size))
        self.first = True

    def _init_params(self,
                     input_: ndarray):
        self.vocab_size = input_.shape[2]
        if not self.weight_scale:
            self.weight_scale = 2 / (self.vocab_size + self.output_size)

        self.params = {
            'W_f': {},
            'B_f': {},
            'W_v': {},
            'B_v': {}
        }
        self.params['W_f']['value'] = np.random.normal(loc=0.0,
                                                       scale=self.weight_scale,
                                                       size=(self.hidden_size + self.vocab_size, self.hidden_size))
        self.params['B_f']['value'] = np.random.normal(loc=0.0,
                                                       scale=self.weight_scale,
                                                       size=(1, self.hidden_size))
        self.params['W_v']['value'] = np.random.normal(loc=0.0,
                                                       scale=self.weight_scale,
                                                       size=(self.hidden_size, self.output_size))
        self.params['B_v']['value'] = np.random.normal(loc=0.0,
                                                       scale=self.weight_scale,
                                                       size=(1, self.output_size))

        self.params['W_f']['deriv'] = np.zeros_like(self.params['W_f']['value'])
        self.params['B_f']['deriv'] = np.zeros_like(self.params['B_f']['value'])
        self.params['W_v']['deriv'] = np.zeros_like(self.params['W_v']['value'])
        self.params['B_v']['deriv'] = np.zeros_like(self.params['B_v']['value'])

        self.cells = [RNNNode() for _ in range(input_.shape[1])]

    def _clear_gradients(self):
        for key in self.params.keys():
            self.params[key]['deriv'] = np.zeros_like(self.params[key]['deriv'])

    def forward(self, x_seq_in: ndarray):
        """
        param x_seq_in: numpy array of shape (batch_size, sequence_length, vocab_size)
        return x_seq_out: numpy array of shape (batch_size, sequence_length, output_size)
        """
        if self.first:
            self._init_params(x_seq_in)
            self.first = False

        batch_size = x_seq_in.shape[0]
        H_in = np.copy(self.start_H)
        H_in = np.repeat(H_in, batch_size, axis=0)
        sequence_length = x_seq_in.shape[1]
        x_seq_out = np.zeros((batch_size, sequence_length, self.output_size))
        for t in range(sequence_length):
            x_in = x_seq_in[:, t, :]
            y_out, H_in = self.cells[t].forward(x_in, H_in, self.params)
            x_seq_out[:, t, :] = y_out
        self.start_H = H_in.mean(axis=0, keepdims=True)
        return x_seq_out

    def backward(self, x_seq_out_grad: ndarray):
        """
        param x_seq_out_grad: numpy array of shape (batch_size, sequence_length, vocab_size)
        return x_seq_in_grad: numpy array of shape (batch_size, sequence_length, vocab_size)
        """
        batch_size = x_seq_out_grad.shape[0]
        sequence_length = x_seq_out_grad.shape[1]
        H_in_grad = np.zeros((batch_size, self.hidden_size))
        x_seq_in_grad = np.zeros((batch_size, sequence_length, self.vocab_size))
        for t in reversed(range(sequence_length)):
            x_out_grad = x_seq_in_grad[:, t, :]
            grad_out, H_in_grad = self.cells[t].backward(x_out_grad, H_in_grad, self.params)
            x_seq_in_grad[:, t, :] = grad_out
        return x_seq_in_grad
