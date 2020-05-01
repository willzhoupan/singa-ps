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
#

from singa import autograd
from singa import module


class CNN(module.Module):

    def __init__(self, num_classes=10, num_channels=1):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.input_size = 28
        self.dimension = 4
        self.conv1 = autograd.Conv2d(num_channels, 20, 5, padding=0)
        self.conv2 = autograd.Conv2d(20, 50, 5, padding=0)
        self.linear1 = autograd.Linear(4 * 4 * 50, 500)
        self.linear2 = autograd.Linear(500, num_classes)
        self.pooling1 = autograd.MaxPool2d(2, 2, padding=0)
        self.pooling2 = autograd.MaxPool2d(2, 2, padding=0)

    def forward(self, x):
        y = self.conv1(x)
        y = autograd.relu(y)
        y = self.pooling1(y)
        y = self.conv2(y)
        y = autograd.relu(y)
        y = self.pooling2(y)
        y = autograd.flatten(y)
        y = self.linear1(y)
        y = autograd.relu(y)
        y = self.linear2(y)
        return y

    def loss(self, out, ty):
        return autograd.softmax_cross_entropy(out, ty)

    def optim(self, loss, dist_option, spars):
        if dist_option == 'fp32':
            self.optimizer.backward_and_update(loss)
        elif dist_option == 'fp16':
            self.optimizer.backward_and_update_half(loss)
        elif dist_option == 'partialUpdate':
            self.optimizer.backward_and_partial_update(loss)
        elif dist_option == 'sparseTopK':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=True,
                                                      spars=spars)
        elif dist_option == 'sparseThreshold':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=False,
                                                      spars=spars)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


def create_model(pretrained=False, **kwargs):
    """Constructs a CNN model.

    Args:
        pretrained (bool): If True, returns a model pre-trained
    """
    model = CNN(**kwargs)

    return model


__all__ = ['CNN', 'create_model']
