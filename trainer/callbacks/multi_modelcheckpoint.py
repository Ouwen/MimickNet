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

class MultiModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, *args, multi_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_models = multi_models
    
    def save_multi_model(self, filepath, weights_only=False):
        for name, model in self.multi_models:
            if weights_only:
                model.save_weights('{}-{}.weights.hdf5'.format(filepath[:-5], name), overwrite=True)
            else:
                model.save('{}-{}.hdf5'.format(filepath[:-5], name), overwrite=True)

    def _save_model(self, epoch, logs):
        """Saves the model.
        Arguments:
                epoch: the epoch this iteration is in.
                logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)

            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning('Can save best model only with %s available, '
                                                    'skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                        ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                                                         current, filepath))
                        self.best = current
                        self.save_multi_model(filepath, weights_only=self.save_weights_only)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                        (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                self.save_multi_model(filepath, weights_only=self.save_weights_only)
            self._maybe_remove_file()