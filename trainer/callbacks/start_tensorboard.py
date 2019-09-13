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
