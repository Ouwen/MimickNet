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

import tarfile
import time
import argparse
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import os

class LogCode(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, code_dir):
        super().__init__()
        self.code_dir = code_dir
        self.log_dir = log_dir
        self.started = False

    def make_tarfile(self, log_dir, code_dir):
        def filter_function(tarinfo):
            if tarinfo.name.endswith('.pyc'):
                return None
            else:
                return tarinfo

        filepath = '{}.tar.gz'.format(str(time.time()), 'w:gz')
        with tarfile.open(filepath, 'w:gz') as tar:
            tar.add(code_dir, arcname=os.path.basename(code_dir), filter=filter_function)
        with file_io.FileIO(filepath, mode='rb') as input_f:
            with file_io.FileIO(os.path.join(log_dir, os.path.basename(filepath)), mode='wb+') as of:
                of.write(input_f.read())
        os.remove(filepath)

    def on_epoch_end(self, *args, **kwargs):
        if not self.started:
            self.make_tarfile(self.log_dir, self.code_dir)
            self.started = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_dir', default='./trainer', help='Forward .hdf5 model filepath')
    parser.add_argument('--log_dir', default='.', help='Code output filepath')
    args = parser.parse_args()

    LogCode.make_tarfile(None, args.log_dir, args.code_dir)
