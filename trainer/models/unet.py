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

class UnetModel(ModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
            
    def __call__(self):
        downsample_path = []
        inputs = tf.keras.layers.Input(shape=self.shape)
        x = inputs
        
        for idx, filter_num in enumerate(self.filters):
            x = self.Conv2D(filter_num)(x)
            x = self.Activation(x)
            x = self.Dropout()(x)
            x = self.Conv2D(filter_num)(x)
            x = self.Activation(x)
            x = self.Dropout()(x)
            if idx != len(self.filters)-1:
                downsample_path.append(x)
                x = tf.keras.layers.MaxPool2D(padding=self.padding)(x)

        downsample_path.reverse()
        reverse_filters = list(self.filters[:-1])
        reverse_filters.reverse()

        # Upsampling path
        for idx, filter_num in enumerate(reverse_filters):
            x = self.Upsample(filter_num)(x)
            x = tf.keras.layers.concatenate([x, downsample_path[idx]])
            x = self.Conv2D(filter_num)(x)
            x = tf.keras.layers.Activation(activation=tf.nn.relu)(x)
            x = self.Dropout()(x)
            x = self.Conv2D(filter_num)(x)
            x = tf.keras.layers.Activation(activation=tf.nn.relu)(x)
            x = self.Dropout()(x)

        x = tf.keras.layers.Conv2D(1, 1)(x)

        return tf.keras.Model(inputs, x)
