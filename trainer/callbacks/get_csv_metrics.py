import argparse
import csv
import tensorflow as tf
import numpy as np
from skimage.transform import match_histograms
from trainer import utils

from multiprocessing import Pool

_FIELDNAMES = ['filename', 'mse', 'mae', 'ssim', 'contrast_structure', 'luminance', 'psnr',
               'iq_mse', 'iq_mae', 'iq_ssim', 'iq_contrast_structure', 'iq_luminance', 'iq_psnr']

class GetCsvMetrics(tf.keras.callbacks.Callback):
    def __init__(self, forward, dataset, job_dir, count):
        super().__init__()
        self.dataset_iterator = iter(dataset)
        self.forward = forward
        self.csv_out_filepath = tf.io.gfile.GFile('{}/metrics.csv'.format(job_dir), 'wb')
        self.count = count
        

    def get_metrics(self, x, y):
        x = tf.constant(x, dtype=tf.float32)
        y = tf.constant(y, dtype=tf.float32)
        psnr = tf.image.psnr(x, y, 1)
        val, cs, l = utils.custom_ssim.ssim(x, y, 1)
        return {
            'mse': tf.math.reduce_mean(tf.math.square((x - y))).numpy(),
            'mae': tf.math.reduce_mean(tf.math.abs((x - y))).numpy(),
            'ssim': val.numpy()[0],
            'contrast_structure': cs.numpy()[0],
            'luminance': l.numpy()[0],
            'psnr': psnr.numpy()[0]
        }
    
    def full_validation(self):
        for i in range(self.count):
            try:
                iq, dtce, params = next(self.dataset_iterator)
                filename = params['filename'].numpy()[0].decode()
                
                iq = iq.numpy()
                dtce = dtce.numpy()

                output = self.forward.predict(iq)
                iq = match_histograms(iq, dtce)
                output = match_histograms(output, dtce)

                dtce_metrics = self.get_metrics(dtce, output)
                iq_metrics = self.get_metrics(iq, output)

                tmp = {}
                for key, value in iq_metrics.items():
                    tmp['iq_{}'.format(key)] = value

                combined = {**tmp, **dtce_metrics}
                combined['filename'] = filename
                self.writer.writerow(combined)
                print('{}/{} - {}'.format(i, self.count, filename))
            except StopIteration:
                print('End of test data')
                break
        
    def on_train_end(self, logs={}):
        print('Running final validation metrics')
        self.writer = csv.DictWriter(self.csv_out_filepath, fieldnames=_FIELDNAMES)
        self.writer.writeheader()
        self.csv_out_filepath.flush()
        self.full_validation()
        self.csv_out_filepath.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--forward', default='./examples/mimicknet.h5', help='Forward .hdf5 model filepath')
    parser.add_argument('--job_dir', default='.', help='Metrics ouptut filepath')
    parser.add_argument('--csv', default='gs://duke-research-us/mimicknet/data/testing-v2.csv')
 
    args = parser.parse_args()
        
    model = tf.keras.models.load_model(args.forward)
    test_dataset, test_count = utils.MimickDataset(
        clipping=(-80,0)
    ).get_paired_ultrasound_dataset(csv=args.csv, batch_size=1)

    get_csv_metrics_data_callback = GetCsvMetrics(model, test_dataset, args.job_dir, count=test_count)
    get_csv_metrics_data_callback.full_validation()
    get_csv_metrics_data_callback.csv_out_filepath.flush()
