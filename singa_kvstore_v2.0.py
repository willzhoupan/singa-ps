from singa import autograd
from singa import tensor
from singa import opt
from singa.proto import core_pb2
import mxnet as mx
import numpy as np
import SingaOpt
import ctypes
#---------start ps API---------------------------------
def create_kvstore(kv_type, opt_type,**kargs):
    kv_ret = mx.kv.create(kv_type) #create kv store for ps
    kv_ret.set_optimizer(mx.optimizer.create(opt_type,**kargs))
    return kv_ret


def tensor2numpy_nocopy(t):
    '''Copy the tensor into a numpy array by sharing the same memory.

    Args:
        t (Tensor): a Tensor

    Returns:
        a numpy array
    '''
    swig_pointer = t.data.GetValueNoCopy()
    ctype_pointer = ctypes.cast(int(swig_pointer),ctypes.POINTER(ctypes.c_float))
    np_array = np.ctypeslib.as_array(ctype_pointer,shape=t.shape)
    return np_array

is_kvInitial = False
#kv_pairs = []
#model_pairs = []
def backward_and_update(kv,loss):
    global is_kvInitial,kv_pairs,model_pairs
    
    if is_kvInitial != True:
       #Initial kv store for workers of ps-architecture
       key = 0
       for p, g in autograd.backward(loss):
           np_p = tensor2numpy_nocopy(p)
           np_g = tensor2numpy_nocopy(g)
           mxnd_p = mx.nd.from_numpy(np_p,zero_copy=True)
           mxnd_g = mx.nd.from_numpy(np_g,zero_copy=True)
           kv.init(key, mxnd_p)
           key += 1  
       is_kvInitial = True
    else:     
       #push
       key = 0
       kv_pairs = []
       model_pairs = []
       #the following push and pull will optimized 
       #according to the performance 
       for p, g in autograd.backward(loss):
           np_p = tensor2numpy_nocopy(p)
           np_g = tensor2numpy_nocopy(g)
           mxnd_p = mx.nd.from_numpy(np_p,zero_copy=True)
           mxnd_g = mx.nd.from_numpy(np_g,zero_copy=True)
           kv.push(key,mxnd_g)
           model_pairs.append((key,p,g))
           kv_pairs.append((key,mxnd_p,mxnd_g))
           key += 1
       #pull
       for key,p,g in model_pairs:
           #np_p = tensor2numpy_nocopy(p)
           #out_buf = mx.nd.from_numpy(np_p,zero_copy=True)
           #out_buf = mx.nd.zeros(p.shape)
           _,out_buf,_ = kv_pairs[key] 
           kv.pull(key,out=out_buf)
           out_buf.wait_to_read()
           #p.copy_from_numpy(out_buf.asnumpy())
           
#--------end ps API---------------------------------

