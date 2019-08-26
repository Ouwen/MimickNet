from tensorflow.python.lib.io import file_io
import os
import tensorflow as tf

class CopyKerasModel(tf.keras.callbacks.Callback):
    def __init__(self, model_dir, job_dir):
        super()
        self.model_dir = model_dir
        self.job_dir = job_dir
        self.model_files = []
    
    def upload_files(self):
        files = [f for f in os.listdir(self.model_dir) if os.path.isfile(os.path.join(self.model_dir, f))]
        for f in files:
            if '.hdf5' in f and f not in self.model_files:
                self.model_files.append(f)
                with file_io.FileIO(os.path.join(self.model_dir, f), mode='rb') as input_f:
                    with file_io.FileIO(os.path.join(self.job_dir, f), mode='wb+') as of:
                        of.write(input_f.read())
    
    def on_epoch_end(self, epoch, logs):
        self.upload_files()
