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
from tensorflow.python.keras import backend as K

class MultiEarlyStopping(tf.keras.callbacks.EarlyStopping):
    def __init__(self, *args, multi_models=None, full_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_models = multi_models
        self.full_model = full_model
        self.multi_best_weights = [model.get_weights() for model in self.multi_models]
        
    def multi_set_best_weights(self):
        self.multi_best_weights = [model.get_weights() for model in self.multi_models]

    def multi_restore_best_weights(self):
        for model, weights in zip(self.multi_models, self.multi_best_weights):
            model.set_weights(weights)

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.multi_set_best_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.full_model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.multi_restore_best_weights()
