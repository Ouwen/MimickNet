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

import argparse
import csv
import tensorflow as tf
import numpy as np
from skimage.transform import match_histograms
from trainer import utils
import sys

from multiprocessing import Pool

_FIELDNAMES = ['filename', 'mse', 'mae', 'ssim', 'contrast_structure', 'luminance', 'psnr', 'corr',
               'iq_mse', 'iq_mae', 'iq_ssim', 'iq_contrast_structure', 'iq_luminance', 'iq_psnr', 'iq_corr',
              'nh_mse', 'nh_mae', 'nh_ssim', 'nh_contrast_structure', 'nh_luminance', 'nh_psnr', 'nh_corr', 
              'nhiq_mse', 'nhiq_mae', 'nhiq_ssim', 'nhiq_contrast_structure', 'nhiq_luminance', 'nhiq_psnr', 'nhiq_corr']

class GetCsvMetrics(tf.keras.callbacks.Callback):
    def __init__(self, forward, dataset, job_dir, count, out='metrics'):
        super().__init__()
        self.dataset_iterator = iter(dataset)
        self.forward = forward
        self.csv_out_filepath = tf.io.gfile.GFile('{}/{}.csv'.format(job_dir, out), 'wb')
        self.count = count
        

    def get_metrics(self, x, y):      
        corr = np.corrcoef(x.flatten(), y.flatten())[0][1]
        
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
            'psnr': psnr.numpy()[0],
            'corr': corr
        }
    
    def full_validation(self):
        for i in range(self.count):
            try:
                iq, dtce, params = next(self.dataset_iterator)
                filename = params['filename'].numpy()[0].decode()
                
                iq = iq.numpy()
                dtce = dtce.numpy()

                output = self.forward.predict(iq)
                output = np.clip(output, 0, 1)
                dtce_metrics = self.get_metrics(dtce, output)
                iq_metrics = self.get_metrics(dtce, iq)

                tmp = {}
                for key, value in iq_metrics.items():
                    tmp['nhiq_{}'.format(key)] = value
                for key, value in dtce_metrics.items():
                    tmp['nh_{}'.format(key)] = value

                dtce = match_histograms(dtce, iq)
                output = match_histograms(output, iq)
                dtce_metrics = self.get_metrics(dtce, output)
                iq_metrics = self.get_metrics(dtce, iq)

                for key, value in iq_metrics.items():
                    tmp['iq_{}'.format(key)] = value

                combined = {**tmp, **dtce_metrics}
                combined['filename'] = filename
                self.writer.writerow(combined)
                print('{}/{} - {}'.format(i, self.count, filename))
            except:
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
    parser.add_argument('--forward', default='./examples/models/mimicknet_1568473738-210304.h5', help='Forward .hdf5 model filepath')
    parser.add_argument('--job_dir', default='.', help='Metrics ouptut filepath')
    parser.add_argument('--csv', default='gs://duke-research-us/mimicknet/data/testing-v2.csv')
    parser.add_argument('--out', default='mimicknet_1568473738-210304_metrics', help='Metrics ouptut filepath')
    parser.add_argument('--clip', default=None)
    
    args = parser.parse_args()
    if args.clip is not None:
        args.clip = (float(args.clip), 0)
        
    model = tf.keras.models.load_model(args.forward)
    test_dataset, test_count = utils.MimickDataset(
        clipping=args.clip
    ).get_paired_ultrasound_dataset(csv=args.csv, batch_size=1)

    get_csv_metrics_data_callback = GetCsvMetrics(model, test_dataset, args.job_dir, count=test_count, out=args.out)
    get_csv_metrics_data_callback.on_train_end()

