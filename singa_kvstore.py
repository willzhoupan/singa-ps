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
#model_pairs = []
def backward_and_update(kv,loss):
    global is_kvInitial
    model_pairs = [] 
    key_list = []
    p_list = []
    if is_kvInitial != True:
       #Initial kv store for workers of ps-architecture
       key = 0
       for p, g in autograd.backward(loss):
           mxnd_p = mx.nd.from_numpy(tensor.to_numpy(p),zero_copy=True)
           kv.init(key, mxnd_p)
           model_pairs.append((key,p,g))
           key += 1  
       is_kvInitial = True
    else:     
       #push
       key = 0
       #the following push and pull will optimized 
       #according to the performance
       for p, g in autograd.backward(loss):
           #create NDarray from p
           #the created NDarray is used to receive pulled parameters with zero copy
           np_p = tensor2numpy_nocopy(p)
           mxnd_p = mx.nd.from_numpy_nocopy(np_p,device_id=p.device.id(),zero_copy=True)
           #copy g to CPU and create NDarray from CPU
           #this can avoid creating memory on GPU0
           g.to_host()
           mxnd_g = mx.nd.from_numpy(tensor2numpy_nocopy(g),zero_copy=True)
           kv.push(key,mxnd_g)
           key_list.append(key)
           p_list.append(mxnd_p)
           model_pairs.append((key,p,g))
           key += 1
       #pull 
       kv.pull(key_list,out=p_list)
       mx.nd.waitall()
       del model_pairs
       del key_list
       del p_list       
#--------end ps API---------------------------------

