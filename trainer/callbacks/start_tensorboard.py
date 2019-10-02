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
import subprocess
import atexit

class StartTensorBoard(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super()
        self.log_dir = log_dir
        self.started = False
        
    def start_tensorboard(self, log_dir):
        try:
            p = subprocess.Popen(['tensorboard', '--logdir', self.log_dir])
        except subprocess.CalledProcessError as err:
            print('ERROR:', err)
            
        atexit.register(lambda: p.kill())    
        print('\n\n Starting Tensorboard at: {}\n\n'.format(self.log_dir))
        
    def on_epoch_end(self, *args, **kwargs):
        if not self.started:
            self.start_tensorboard(self.log_dir)
            self.started = True
