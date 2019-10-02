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
import argparse
import numpy as np

class GenerateImages(tf.keras.callbacks.Callback):
    def __init__(self, forward, dataset, log_dir, interval=1000, postfix='val'):
        super()
        self.step_count = 0
        self.postfix = postfix
        self.interval = interval
        self.forward = forward
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.dataset_iterator = iter(dataset)

    def generate_images(self):
        iq, dtce = next(self.dataset_iterator)
        fake_dtce = self.forward.predict(iq)
        fake_dtce = tf.clip_by_value(fake_dtce, 0, 1).numpy()
        
        with self.summary_writer.as_default():
            tf.summary.image('{}/mimicknet'.format(self.postfix), fake_dtce, step=self.step_count)
            tf.summary.image('{}/clinical'.format(self.postfix), dtce, step=self.step_count)
            tf.summary.image('{}/das'.format(self.postfix), iq, step=self.step_count)
            tf.summary.image('{}/delta'.format(self.postfix), tf.abs(fake_dtce-dtce), step=self.step_count)

    def on_batch_begin(self, batch, logs={}):
        self.step_count += 1
        if self.step_count % self.interval == 0:
            self.generate_images()
            
    def on_train_end(self, logs={}):
        self.generate_images()
