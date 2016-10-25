from chainer import function
from chainer.utils import type_check
import cupy

def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)

time_span = 10


class LinearL4Function(function.Function):
    '''
    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim >= 2,
            w_type.ndim == 2,
            type_check.prod(x_type.shape[1:]) == w_type.shape[1],
        )
        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )
    '''
    def forward(self, inputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        #for MQN, split x
        count_x = len(x[0]) / (time_span + 1) #expected 10,240
        x_now = x[:,:count_x]
        #print 'L4 x_now.shape=',
        #print x_now.shape
        #print 'L4 W.shape=',
        #print W.shape        

        y = x_now.dot(W.T).astype(x.dtype, copy=False)
        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y,

    def backward(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        #for MQN, split x
        count_x = len(x[0]) / (time_span + 1) #expected 10,240
        x_now = x[:,:count_x]
        #print 'L4 x_now.shape=',
        #print x_now.shape

        W = inputs[1]
        gy = grad_outputs[0]

        gx = gy.dot(W).astype(x.dtype, copy=False)
        #print 'L4 gx.shape=',
        #print gx.shape

        #for MQN, tile gx
        gx_tile = cupy.tile(gx, (1,(time_span + 1))).astype(x.dtype, copy=False)
        #print 'L4 gx_tile.shape=',
        #print gx_tile.shape
        gx_reshape = gx_tile.reshape(inputs[0].shape)
        #print 'L4 gx_reshape.shape=',
        #print gx_reshape.shape

        gW = gy.T.dot(x_now).astype(W.dtype, copy=False)

        if len(inputs) == 3:
            gb = gy.sum(0)
            return gx_reshape, gW, gb
        else:
            return gx_reshape, gW


def linearL4_func(x, W, b=None):
    """Linear function, or affine transformation.

    It accepts two or three arguments: an input minibatch ``x``, a weight
    matrix ``W``, and optionally a bias vector ``b``. It computes
    :math:`Y = xW^\\top + b`.

    Args:
        x (~chainer.Variable): Input variable. Its first dimension is assumed
            to be the *minibatch dimension*. The other dimensions are treated
            as concatenated one dimension whose size must be ``N``.
        W (~chainer.Variable): Weight variable of shape ``(M, N)``.
        b (~chainer.Variable): Bias variable (optional) of shape ``(M,)``.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`~chainer.links.Linear`

    """
    if b is None:
        return LinearL4Function()(x, W)
    else:
        return LinearL4Function()(x, W, b)
