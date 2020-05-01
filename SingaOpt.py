import mxnet as mx
import numpy
import copy
from mxnet.ndarray import NDArray
from singa import opt
from singa import tensor

from singa.proto import core_pb2
def tensor2numpy_nocopy(th):
    '''Copy the tensor into a numpy array by sharing the same memory.

    Args:
        t (Tensor): a Tensor

    Returns:
        a numpy array
    '''
    if th.dtype == core_pb2.kFloat32:
        np_array = th.data.GetFloatValue(int(th.size()))
    elif th.dtype == core_pb2.kInt:
        np_array = th.data.GetIntValue(int(th.size()))
    else:
        print('Not implemented yet for ', th.dtype)
    return np_array.reshape(th.shape)


@mx.optimizer.Optimizer.register
class SingaSGD(mx.optimizer.Optimizer):
    """The Test optimizer"""
    def __init__(self, **kwargs):
        super(SingaSGD, self).__init__()
        self.sgd = opt.SGD(**kwargs)
        
    def create_state(self, index, weight):
        """Creates a state to duplicate weight."""
        return mx.ndarray.zeros(weight.shape, weight.context)

    def step(self, indices, weights, grads, states):
        """Performs w += rescale_grad * grad."""
        if type(indices).__name__ == 'int':
           indices = [indices]
           weights = [weights]
           grads = [grads]

        for index, weight, grad in zip(indices, weights, grads):
            p = tensor.Tensor(shape=weight.shape,
                              #device=weight.context,
                              #dtype=weight.dtype,
                              data=weight.asnumpy())
            g = tensor.Tensor(shape=grad.shape,
                              #device=grad.context,
                              #dtype=grad.dtype,
                              data=grad.asnumpy())
            self.sgd.update(p,g)
            weight[:] = tensor.to_numpy(p)
            #weight[:] = tensor2numpy_nocopy(p) #nocopy

    def update(self, indices, weights, grads, states):
        """Call step to perform a single optimization update if use_fused_step is False,
         otherwise fused_step is called.

        Parameters
        ----------
        indices : list of int
            List of unique indices of the parameters into the individual learning rates
            and weight decays. Learning rates and weight decay may be set via `set_lr_mult()`
            and `set_wd_mult()`, respectively.
        weights : list of NDArray
            List of parameters to be updated.
        grads : list of NDArray
            List of gradients of the objective with respect to this parameter.
        states : List of any obj
            List of state returned by `create_state()`.
        """
        for weight, grad in zip(weights, grads):
            assert(isinstance(weight, NDArray))
            assert(isinstance(grad, NDArray))
        self.step(indices, weights, grads, states)

