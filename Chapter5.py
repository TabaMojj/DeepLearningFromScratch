import numpy as np
from numpy import ndarray

from Chapter3 import ParamOperation


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


class Conv2D_Op(ParamOperation):

    def __init__(self, W: ndarray):
        super().__init__(W)
        self.param_size = W.shape[2]
        self.param_pad = self.param_size // 2

    def _pad_1d(self, inp: ndarray) -> ndarray:
        z = np.array([0])
        z = np.repeat(z, self.param_pad)
        return np.concatenate([z, inp, z])

    def _pad_1d_batch(self,
                      inp: ndarray) -> ndarray:
        outs = [self._pad_1d(obs) for obs in inp]
        return np.stack(outs)

    def _pad_2d_obs(self,
                    inp: ndarray):
        '''
        Input is a 2 dimensional, square, 2D Tensor
        '''
        inp_pad = self._pad_1d_batch(inp)

        other = np.zeros((self.param_pad, inp.shape[0] + self.param_pad * 2))

        return np.concatenate([other, inp_pad, other])

    # def _pad_2d(self,
    #             inp: ndarray):
    #     '''
    #     Input is a 3 dimensional tensor, first dimension batch size
    #     '''
    #     outs = [self._pad_2d_obs(obs, self.param_pad) for obs in inp]
    #
    #     return np.stack(outs)

    def _pad_2d_channel(self,
                        inp: ndarray):
        '''
        inp has dimension [num_channels, image_width, image_height]
        '''
        return np.stack([self._pad_2d_obs(channel) for channel in inp])

    def _get_image_patches(self,
                           input_: ndarray):
        imgs_batch_pad = np.stack([self._pad_2d_channel(obs) for obs in input_])
        patches = []
        img_height = imgs_batch_pad.shape[2]
        for h in range(img_height - self.param_size + 1):
            for w in range(img_height - self.param_size + 1):
                patch = imgs_batch_pad[:, :, h:h + self.param_size, w:w + self.param_size]
                patches.append(patch)
        return np.stack(patches)

    def _output(self,
                inference: bool = False):
        '''
        conv_in: [batch_size, channels, img_width, img_height]
        param: [in_channels, out_channels, fil_width, fil_height]
        '''
        #     assert_dim(obs, 4)
        #     assert_dim(param, 4)
        batch_size = self.input_.shape[0]
        img_height = self.input_.shape[2]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        patch_size = self.param.shape[0] * self.param.shape[2] * self.param.shape[3]

        patches = self._get_image_patches(self.input_)

        patches_reshaped = (patches
                            .transpose(1, 0, 2, 3, 4)
                            .reshape(batch_size, img_size, -1))

        param_reshaped = (self.param
                          .transpose(0, 2, 3, 1)
                          .reshape(patch_size, -1))

        output_reshaped = (
            np.matmul(patches_reshaped, param_reshaped)
            .reshape(batch_size, img_height, img_height, -1)
            .transpose(0, 3, 1, 2))

        return output_reshaped

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:

        batch_size = self.input_.shape[0]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        img_height = self.input_.shape[2]

        output_patches = (self._get_image_patches(output_grad)
                          .transpose(1, 0, 2, 3, 4)
                          .reshape(batch_size * img_size, -1))

        param_reshaped = (self.param
                          .reshape(self.param.shape[0], -1)
                          .transpose(1, 0))

        return (
            np.matmul(output_patches, param_reshaped)
            .reshape(batch_size, img_height, img_height, self.param.shape[0])
            .transpose(0, 3, 1, 2)
        )

    def _param_grad(self, output_grad: ndarray) -> ndarray:

        batch_size = self.input_.shape[0]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        in_channels = self.param.shape[0]
        out_channels = self.param.shape[1]

        in_patches_reshape = (
            self._get_image_patches(self.input_)
            .reshape(batch_size * img_size, -1)
            .transpose(1, 0)
        )

        out_grad_reshape = (output_grad
                            .transpose(0, 2, 3, 1)
                            .reshape(batch_size * img_size, -1))

        return (np.matmul(in_patches_reshape,
                          out_grad_reshape)
                .reshape(in_channels, self.param_size, self.param_size, out_channels)
                .transpose(0, 3, 1, 2))
