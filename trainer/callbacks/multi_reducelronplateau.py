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

class MultiReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(self, *args, training_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_models = training_models
    
    def set_new_lr(self, new_lr):
        for model in self.training_models:
            K.set_value(model.optimizer.lr, new_lr)
    
    def get_lr(self):
        return K.get_value(self.training_models[0].optimizer.lr)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = self.get_lr()
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                                            'which is not available. Available metrics are: %s',
                                            self.monitor, ','.join(list(logs.keys())))

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(self.get_lr())
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        self.set_new_lr(new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                        'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
