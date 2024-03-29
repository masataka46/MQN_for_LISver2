import math

#from chainer.functions.connection import linear
from chainer import initializers
from chainer import link
import linearL4_func

time_span = 10

class LinearL4_link(link.Link):

    """Linear layer (a.k.a. fully-connected layer).

    This is a link that wraps the :func:`~chainer.functions.linear` function,
    and holds a weight matrix ``W`` and optionally a bias vector ``b`` as
    parameters.

    The weight matrix ``W`` is initialized with i.i.d. Gaussian samples, each
    of which has zero mean and deviation :math:`\\sqrt{1/\\text{in_size}}`. The
    bias vector ``b`` is of size ``out_size``. Each element is initialized with
    the ``bias`` value. If ``nobias`` argument is set to True, then this link
    does not hold a bias vector.

    Args:
        in_size (int): Dimension of input vectors. If None, parameter
            initialization will be deferred until the first forward data pass
            at which time the size will be determined.
        out_size (int): Dimension of output vectors.
        wscale (float): Scaling factor of the weight matrix.
        bias (float): Initial bias value.
        nobias (bool): If ``True``, then this function does not use the bias.
        initialW (2-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.

    .. seealso:: :func:`~chainer.functions.linear`

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.

    """

    def __init__(self, in_size, out_size, wscale=1, bias=0, nobias=False,
                 initialW=None, initial_bias=None):
        super(LinearL4_link, self).__init__()
        self.initialW = initialW
        self.wscale = wscale
        self.out_size = out_size
        self.in_size = in_size / (time_span + 1)

        if in_size is None:
            self.add_uninitialized_param('W')
        else:
            self._initialize_params(self.in_size)

        if nobias:
            self.b = None
        else:
            self.add_param('b', out_size)
            if initial_bias is None:
                initial_bias = bias
            initializers.init_weight(self.b.data, initial_bias)

    def _initialize_params(self, in_size):
        self.add_param('W', (self.out_size, in_size))
        # For backward compatibility, the scale of weights is proportional to
        # the square root of wscale.
        initializers.init_weight(self.W.data, self.initialW,
                                 scale=math.sqrt(self.wscale))

    def __call__(self, x):
        """Applies the linear layer.

        Args:
            x (~chainer.Variable): Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the linear layer.

        """
        if self.has_uninitialized_params:
            self._initialize_params(x.shape[1])
        return linearL4_func.linearL4_func(x, self.W, self.b)
