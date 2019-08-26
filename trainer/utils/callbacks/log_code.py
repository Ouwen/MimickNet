import tarfile
import time
import argparse
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import os

class LogCode(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, code_dir):
        super()
        self.code_dir = code_dir
        self.log_dir = log_dir
        
    def make_tarfile(self, log_dir, code_dir):
        def exclude_function(filename):
            return True if filename.endswith('.pyc') else False
        
        filepath = '{}.tar.gz'.format(str(time.time()), 'w:gz')
        with tarfile.open(filepath, 'w:gz') as tar:
            tar.add(code_dir, arcname=os.path.basename(code_dir), exclude=exclude_function)
        with file_io.FileIO(filepath, mode='rb') as input_f:
            with file_io.FileIO(os.path.join(log_dir, os.path.basename(filepath)), mode='wb+') as of:
                of.write(input_f.read())
        os.remove(filepath)

    def on_train_begin(self, logs={}):
        self.make_tarfile(self.log_dir, self.code_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_dir', default='./trainer', help='Forward .hdf5 model filepath')
    parser.add_argument('--log_dir', default='.', help='Code output filepath')
    args = parser.parse_args()

    LogCode.make_tarfile(None, args.log_dir, args.code_dir)
