from chainer import function
from chainer.utils import type_check
import numpy
import cupy


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)
time_span = 10

class memory_unit_Function(function.Function):
    #time_span = 10
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
        #print 'L56 x.shape=',
        #print x.shape
        #print 'L56 W.shape=',
        #print W.shape
        '''
        #modified for P91
        x_ext = cupy.tile(x, (1, 10)).astype(x.dtype, copy=False)
        #print 'L56 x_ext.shape=',
        #print x_ext.shape
        '''
        #operation by which 11times change to 10times
        count_x = len(x[0]) / (time_span + 1) #expected 10,240
        x_time = x[:,count_x:]
        #print 'L56 x_time.shape=',
        #print x_time.shape 
        
        #reshape x
        x_reshape = x_time.reshape(len(x_time), time_span, -1)
        #print 'L56 x_reshape.shape =',
        #print x_reshape.shape
        #print 'L56 W.T.shape =',
        #print W.T.shape
        
        y = x_reshape.dot(W.T).astype(x.dtype, copy=False)
        '''
        if len(inputs) == 3:
            b = inputs[2]
            y += b
        '''
        y_flat = y.reshape(len(x), -1)
        #y_flat2 = y_flat.reshape(len(y_flat), -1)
        #print 'L56 y_flat.shape',
        #print y_flat.shape
        #print 'L56 len(y_flat[0])',
        #print len(y_flat[0])

        return y_flat,

    def backward(self, inputs, grad_outputs):
        #print 'L56 now backward'
        x = _as_mat(inputs[0])
        W = inputs[1]
        gy = grad_outputs[0]
        #print 'L56 backward'
        #print 'L56 gy.shape=',
        #print gy.shape

        #modified 16/10/20
        gy_cube = gy.reshape(len(gy), time_span, -1).astype(dtype=gy.dtype, copy=False)
        #print 'L56 gy_cube.shape=',
        #print gy_cube.shape

        gx_cube = gy_cube.dot(W).astype(dtype=gy.dtype, copy=False)
        #print 'L56 gx_cube.shape',
        #print gx_cube.shape
        '''
        #modified for P91
        gx_sum = cupy.sum(gx_cube, axis=1).astype(dtype=gy.dtype, copy=False)
        #print 'L56 gx_sum.shape',
        #print gx_sum.shape
        '''
        #operation by which 11times change to 10times
        count_x = len(x[0]) / (time_span + 1) #expected 10,240
        x_time = x[:,count_x:]

        #print 'L56 x_time.shape=',
        #print x_time.shape

        gx = gx_cube.reshape(len(gx_cube), -1).astype(dtype=gy.dtype, copy=False)
        #print 'L56 gx.shape=',
        #print gx.shape
        gx_plus = cupy.zeros((len(gx), len(gx[0]) / time_span), dtype=gx.dtype)
        #print 'L56 gx_plus.shape=',
        #print gx_plus.shape
        gx_long = cupy.concatenate([gx, gx_plus], axis=1).astype(dtype=gy.dtype, copy=False)
        #print 'L56 gx_long.shape=',
        #print gx_long.shape
        gx_reshape = gx_long.reshape(inputs[0].shape)   

        '''
        gx = gx_sum.reshape(inputs[0].shape)
        #print 'L56 gx.shape',
        #print gx.shape
        '''
        gy_sum = cupy.sum(gy_cube, axis=0).astype(dtype=gy.dtype, copy=False)
        #print 'L56 gy_sum.shape',
        #print gy_sum.shape
        '''
        #modified for P91 (extent 10,240 to 102,400)
        x_ext = cupy.tile(x, (1, 10)).astype(x.dtype, copy=False)
        #print 'L56 x_ext.shape=',
        #print x_ext.shape
        '''
        
        
        x_reshape = x_time.reshape(len(x_time), time_span, -1)
        #print 'L56 x_reshape.shape',
        #print x_reshape.shape

        x_sum = cupy.sum(x_reshape, axis=0).astype(dtype=gy.dtype, copy=False)
        #print 'L56 x_sum.shape',
        #print x_sum.shape

        gW = gy_sum.T.dot(x_sum).astype(dtype=gy.dtype, copy=False)
        #print 'L56 gW.shape',
        #print gW.shape

        #make x(n,time_span,value)
        #x_span = cypy.hsplit(x, time_span)

        #x_sum = numpy.zeros(len(x_span[0]), len(x_span[0][0])).astype(x.dtype, copy=False)
        #x_sum = numpy.zeros((len(x_span[0]), len(x_span[0][0])), dtype=x.dtype)

        '''
        for i in range(time_span):
            x_sum = x_sum + x_span[i]
        
        gy_sum = numpy.zeros((len(gy), len(gy[0]) / time_span), dtype=gy.dtype)
        gy_span = numpy.hsplit(gy, time_span)
        
        ##print 'L56 len(gy_sum)=',
        ##print len(gy_sum)
        ##print 'L56 len(gy_sum[0])=',
        ##print len(gy_sum[0])
        ##print 'len(gy_span)=',
        ##print len(gy_span)
        ##print 'len(gy_span[0])=',
        ##print len(gy_span[0])
        
        for i in range(time_span):
            gy_sum = gy_sum + gy_span[i]
        
        #gx = gy.dot(W).astype(x.dtype, copy=False).reshape(inputs[0].shape)
        gx = gy_sum.dot(W).astype(x.dtype, copy=False)
        ##print 'len(gx)=',
        ##print len(gx)
        ##print 'len(gx[0])=',
        ##print len(gx[0])

        #make gx_ext(n, input[0])
        gx_cop = numpy.zeros((time_span, len(gx), len(gx[0])), dtype=gx.dtype)
        for i in range(time_span):
            gx_cop[i] = gx
        ##print 'len(gx_cop)=',
        ##print len(gx_cop)
        ##print 'len(gx_cop[0])=',
        ##print len(gx_cop[0])
        gx_ext = gx_cop[0]
        for i in range(time_span - 1):
            gx_ext = numpy.concatenate((gx_ext, gx_cop[i+1]), axis=1)

        ##print 'len(gx_cop)=',
        ##print len(gx_cop)
        ##print 'len(gx_cop[0])=',
        ##print len(gx_cop[0])

        gW = gy_sum.T.dot(x_sum).astype(W.dtype, copy=False)
        ##print 'len(gW)=',
        ##print len(gW)
        ##print 'len(gW[0])=',
        ##print len(gW[0])

        ##print 'len(gx_ext)=',
        ##print len(gx_ext)
        ##print 'len(gx_ext[0])=',
        ##print len(gx_ext[0])
        
        gx_ext2 = gx_ext.reshape(inputs[0].shape)

        ##print 'len(gx_ext2)=',
        ##print len(gx_ext2)
        ##print 'len(gx_ext2[0])=',
        ##print len(gx_ext2[0])
        
        '''
        if len(inputs) == 3:
            gb = gy.sum(0)
            #print'gb.shape=',
            #print gb.shape
            return gx_reshape, gW, gb
        else:
            return gx_reshape, gW


def memory_unit_func(x, W, time_span, b=None):
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
        return memory_unit_Function()(x, W)
    else:
        return memory_unit_Function()(x, W, b)
