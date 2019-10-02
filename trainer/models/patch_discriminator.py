"""
Copyright Ouwen Huang 2019 

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf
from trainer.models.model_base import ModelBase

class PatchDiscriminatorModel(ModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __call__(self):
        inputs = tf.keras.Input(shape=self.shape)
        x = inputs
        for filter_num in self.filters:
            x = self.Conv2D(filter_num)(x)
            x = self.Activation(x)
            x = tf.keras.layers.MaxPool2D(padding=self.padding)(x)
        validity = tf.keras.layers.Conv2D(1, (self.shape[0]//(2**(len(self.filters)+1)), 
                                              self.shape[1]//(2**(len(self.filters)+1))), padding=self.padding)(x)
        return tf.keras.Model(inputs, validity)
