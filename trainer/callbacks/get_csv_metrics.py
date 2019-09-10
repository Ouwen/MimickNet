import argparse
import csv
import pandas as pd
import tensorflow as tf
import numpy as np
from trainer.utils.dataset import loadmat, make_shape, scan_convert
from trainer.utils import custom_ssim

class GetCsvMetrics(tf.keras.callbacks.Callback):
    def __init__(self, forward, job_dir, log_compress = True, clipping = -80,
                 test_csv='gs://duke-research-us/mimicknet/data/testing-v1.csv', 
                 bucket_dir='gs://duke-research-us/mimicknet/data/duke-ultrasound-v1', 
                 image_dir=None):
        super()
        self.bucket_dir = bucket_dir
        self.image_dir = image_dir
        self.filelist = list(pd.read_csv(tf.io.gfile.GFile(test_csv, 'rb'))['filename'])
        self.forward = forward
        self.log_compress = log_compress
        self.clipping =  clipping

        self.csv_filepath = tf.io.gfile.GFile('{}/metrics.csv'.format(job_dir), 'wb')
        self.writer = csv.DictWriter(self.csv_filepath, fieldnames=['filename', 'mse', 'mae', 'ssim', 'contrast_structure', 'luminance', 'psnr'])
        self.writer.writeheader()
        self.csv_filepath.flush()

    def get_metrics(self, x, y):
        psnr = tf.image.psnr(x, y)
        val, cs, l = custom_ssim.ssim(x, y, 1)
        return {
            'mse': tf.math.reduce_mean(tf.math.square((x - y))).numpy(),
            'mae': tf.math.reduce_mean(tf.math.abs((x - y))).numpy(),
            'ssim': val.numpy()[0],
            'contrast_structure': cs.numpy()[0],
            'luminance': l.numpy()[0],
            'psnr': psnr.numpy()[0]
        }
    
    def full_validation(self):
        for i, filename in enumerate(self.filelist):
            print('{}/{}, {}'.format(i, len(self.filelist), filename))
            filepath = tf.io.gfile.GFile('{}/{}'.format(self.bucket_dir, filename), 'rb')
            if self.image_dir is not None: 
                filepath = '{}/{}'.format(self.image_dir, filename)

            matfile = loadmat(filepath)
            iq = abs(matfile['iq'])
            if self.clipping is not None:
                iq = 20*np.log10(iq/iq.max())
                iq = np.clip(iq, self.clipping, 0)
            elif self.log_compress:
                iq = np.log10(iq)
            iq = (iq-iq.min())/(iq.max() - iq.min())
            dtce = matfile['dtce']
            dtce = (dtce - dtce.min())/(dtce.max() - dtce.min())

            iq, _ = make_shape(iq)
            dtce, _ = make_shape(dtce)
            iq = np.expand_dims(np.expand_dims(iq, axis=0), axis=-1)
            dtce = np.expand_dims(np.expand_dims(dtce, axis=0), axis=-1)

            acq_params = matfile['acq_params']

            output = self.forward.predict(iq)

            output = scan_convert(np.squeeze(output), acq_params)
            output = np.clip(output, 0, 1)
            output = np.expand_dims(np.expand_dims(output, axis=0), axis=-1)

            dtce = scan_convert(np.squeeze(dtce), acq_params)
            dtce = np.clip(dtce, 0, 1)
            dtce = np.expand_dims(np.expand_dims(dtce, axis=0), axis=-1)          
            
            my_metrics = get_metrics(dtce, output)
            my_metrics['filename'] = filename
            self.writer.writerow(my_metrics)
            
    def on_train_end(self, logs={}):
        print('Running final validation metrics')
        self.full_validation()
        self.csv_filepath.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--forward', default=None, help='Forward .hdf5 model filepath')
    parser.add_argument('--job_dir', default='.', help='Metrics ouptut filepath')
    parser.add_argument('--lg_c',     default= True,  type=bool, help='Log compress the raw IQ data')
    parser.add_argument('--clip',     default= True,  type=bool, help='Clip to -80 of raw beamformed data')
    parser.add_argument('--bucket_dir', default='gs://duke-research-us/mimicknet/data/duke-ultrasound-v1', help='Job directory for Google Cloud ML')
    parser.add_argument('--image_dir', default=None)
    parser.add_argument('--test_csv', default='gs://duke-research-us/mimicknet/data/testing-v1.csv')
    parser.add_argument('--force', action='store_true')
    
    args = parser.parse_args()
    
    if args.forward is None:
        raise ValueError("{} is not valid directory".format(args.forward))
    
    def randomfunc(x,y): return tf.image.ssim(x,y, max_val=1)
    model = tf.keras.models.load_model(args.forward, custom_objects={'ssim_loss': randomfunc,
                                                                     'ssim': randomfunc,
                                                                     'ssim_multiscale': randomfunc,
                                                                     'psnr':randomfunc})
    
    get_csv_metrics_data_callback = GetCsvMetrics(model, args.job_dir, 
                                           log_compress = args.lg_c, 
                                           clipping = args.clip, 
                                           bucket_dir=args.bucket_dir, 
                                           image_dir=args.image_dir, 
                                           test_csv=args.test_csv)
    get_csv_metrics_data_callback.full_validation()
    get_csv_metrics_data_callback.csv_filepath.flush()
