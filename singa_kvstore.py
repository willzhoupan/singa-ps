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
key_list = [] 
value_list = []
#model_pairs = []
def backward_and_update(kv,loss):
    global is_kvInitial,key_list,value_list
    model_pairs = [] 
    if is_kvInitial != True:
       #Initial kv store for workers of ps-architecture
       key = 0
       for p, g in autograd.backward(loss):
           np_p = tensor2numpy_nocopy(p)
           mxnd_p = mx.nd.from_numpy(np_p,device_id=p.device.id(),zero_copy=True)
           #kv_pairs.append((key,mxnd_p))
           key_list.append(key)
           value_list.append(mxnd_p)
           kv.init(key, mxnd_p)
           key += 1  
       is_kvInitial = True
    else:     
       #push
       key = 0
       #the following push and pull will optimized 
       #according to the performance
       for p, g in autograd.backward(loss):
           np_g = tensor2numpy_nocopy(g)
           mxnd_g = mx.nd.from_numpy(np_g,device_id=g.device.id(),zero_copy=True)
           kv.push(key,mxnd_g)
           model_pairs.append((key,p,g,mxnd_g))
           key += 1
       #pull 
       kv.pull(key_list,out=value_list)
       for item in value_list:
           item.wait_to_read()
       del model_pairs  

def backward_and_update_cpu(kv,loss):
    global is_kvInitial,key_list,value_list
    model_pairs = [] 
    if is_kvInitial != True:
       #Initial kv store for workers of ps-architecture
       key = 0
       for p, g in autograd.backward(loss):
           np_p = tensor2numpy_nocopy(p)
           mxnd_p = mx.nd.from_numpy(np_p,zero_copy=True)
           #kv_pairs.append((key,mxnd_p))
           key_list.append(key)
           value_list.append(mxnd_p)
           kv.init(key, mxnd_p)
           key += 1  
       is_kvInitial = True
    else:     
       #push
       key = 0
       #the following push and pull will optimized 
       #according to the performance
       for p, g in autograd.backward(loss):
           np_g = tensor2numpy_nocopy(g)
           mxnd_g = mx.nd.from_numpy(np_g,zero_copy=True)
           kv.push(key,mxnd_g)
           model_pairs.append((key,p,g,mxnd_g))
           key += 1
       #pull 
       kv.pull(key_list,out=value_list)
       for item in value_list:
           item.wait_to_read()
       del model_pairs       
#--------end ps API---------------------------------

