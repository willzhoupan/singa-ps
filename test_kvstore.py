#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
import numpy as np
import numpy.random as rnd
import copy
import MyOpt


keys = ['1', '2', '3']
# let the last shape exceed MXNET_KVSTORE_BIGARRAY_BOUND
shapes = [(1, 1), (2, 2), (3, 3)]
kv_type = 'dist_sync'
lr = .1
nworker = 1
repeat = 1
 
# generate data
data = [np.ones(shapes[i])*2 for i in range(len(keys))]
data3 = [mx.nd.ones(shapes[i])*3 for i in range(len(keys))]
print('gerated data')
print(data)
kv = mx.kv.create(kv_type)
kv.set_optimizer(mx.optimizer.create('sgd',learning_rate=0.1))

for i in range(len(keys)):
    kv.init(keys[i], mx.nd.array(data[i]))

#kv.init(keys,data3)
for r in range(repeat):
    for i in range(len(keys)):
        kv.push(keys[i],mx.nd.array(data[i]))
        out_buf = mx.nd.zeros(shapes[i])
        kv.pull(keys[i],out=out_buf)
        print(out_buf.asnumpy())


    


