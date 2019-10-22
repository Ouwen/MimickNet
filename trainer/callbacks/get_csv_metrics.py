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
import pandas as pd
import scipy.io as sio

from multiprocessing import Pool

_FIELDNAMES = ['filename', 'mse', 'mae', 'ssim', 'contrast_structure', 'luminance', 'psnr', 'corr',
               'iq_mse', 'iq_mae', 'iq_ssim', 'iq_contrast_structure', 'iq_luminance', 'iq_psnr', 'iq_corr',
              'hm_mse', 'hm_mae', 'hm_ssim', 'hm_contrast_structure', 'hm_luminance', 'hm_psnr', 'hm_corr',
              'hm_iq_mse', 'hm_iq_mae', 'hm_iq_ssim', 'hm_iq_contrast_structure', 'hm_iq_luminance', 'hm_iq_psnr', 'hm_iq_corr']


class GetCsvMetrics(tf.keras.callbacks.Callback):
    def __init__(self, forward, dataset_csvpath, job_dir, out='metrics',
                 bucket='gs://duke-research-us/mimicknet/data/duke-ultrasound-v1', clip=-80):
        super().__init__()
        self.df = pd.read_csv(tf.io.gfile.GFile(dataset_csvpath, 'rb'))
        self.forward = forward
        self.csv_out_filepath = tf.io.gfile.GFile('{}/{}.csv'.format(job_dir, out), 'wb')
        self.bucket = bucket
        self.clip = clip
        
    def get_metrics(self, x, y):
        corr = np.corrcoef(x.flatten(), y.flatten())[0][1]
        
        x = tf.constant(x[:,:,None], dtype=tf.float32)
        y = tf.constant(y[:,:,None], dtype=tf.float32)
        
        psnr = tf.image.psnr(x, y, 1)
        val, cs, l = utils.custom_ssim.ssim(x, y, 1)

        return {
            'mse': tf.math.reduce_mean(tf.math.square((x - y))).numpy(),
            'mae': tf.math.reduce_mean(tf.math.abs((x - y))).numpy(),
            'ssim': val.numpy(),
            'contrast_structure': cs.numpy(),
            'luminance': l.numpy(),
            'psnr': psnr.numpy(),
            'corr': corr
        }

    def get_ele(self, ele, iq_key='iq', dtce_key='dtce'):
        # Get images
        matfile = sio.loadmat(tf.io.gfile.GFile('{}/{}'.format(self.bucket, ele['filename']), 'rb'))
        iq = np.abs(matfile[iq_key])
        iq = 20*np.log10(iq/iq.max())
        iq = np.clip(iq, self.clip, 0)
        iq = (iq - iq.min())/(iq.max()-iq.min())
        dtce = (matfile[dtce_key] - matfile[dtce_key].min())/(matfile[dtce_key].max()-matfile[dtce_key].min())
        ele['das'] = iq
        ele['dtce'] = dtce
        output = self.forward.predict(iq[None, :, :, None])
        ele['output'] = np.squeeze(output[0])
        return ele

    def full_validation(self):
        for i, row in self.df.iterrows():
            ele = self.get_ele(row)
            tmp = {}
            dtce_metrics = self.get_metrics(ele['dtce'], ele['output'])
            iq_metrics = self.get_metrics(ele['dtce'], ele['das'])
            for key, value in iq_metrics.items():
                tmp['iq_{}'.format(key)] = value
            for key, value in dtce_metrics.items():
                tmp['{}'.format(key)] = value

            dtce = match_histograms(ele['dtce'], ele['das'])
            output = match_histograms(ele['output'], ele['das'])
            dtce_metrics = self.get_metrics(dtce, output)
            iq_metrics = self.get_metrics(dtce, output)
            for key, value in iq_metrics.items():
                tmp['hm_iq_{}'.format(key)] = value
            for key, value in dtce_metrics.items():
                tmp['hm_{}'.format(key)] = value
                

            tmp['filename'] = row.filename
            self.writer.writerow(tmp)
            print('{}/{} - {}'.format(i, len(self.df), row.filename))

    def on_train_end(self, logs={}):
        print('Running final validation metrics')
        self.writer = csv.DictWriter(self.csv_out_filepath, fieldnames=_FIELDNAMES)
        self.writer.writeheader()
        self.csv_out_filepath.flush()
        self.full_validation()
        self.csv_out_filepath.flush()
        
def custom_pad(inputs):
    shape = tf.shape(inputs)
    height_pad = (16 - shape[1] % 16)
    width_pad = (16 - shape[2] % 16)
    padded = tf.pad(inputs, [[0,0],
                           [0,height_pad],
                           [0,width_pad],
                           [0,0]], mode='REFLECT')
    return padded

def custom_depad(inputs):
    original = inputs[0]
    final = inputs[1]
    shape = tf.shape(original)
    final = final[:, :shape[1], :shape[2], :]
    final = tf.clip_by_value(final, 0, 1)
    return final

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--forward', default='./examples/models/mimicknet_1568473738-210304.h5',
                        help='Forward .hdf5 model filepath')
    parser.add_argument('--job_dir', default='.', help='Metrics ouptut filepath')
    parser.add_argument('--csv', default='gs://duke-research-us/mimicknet/data/testing-v2.csv')
    parser.add_argument('--out', default='mimicknet_1568473738-210304_metrics', help='Metrics ouptut filepath')
    parser.add_argument('--clip', type=int, default=-80)
    
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.forward)
    inputs = tf.keras.layers.Input(shape=(None, None, 1))
    x = tf.keras.layers.Lambda(custom_pad)(inputs)
    x = model(x)
    x = tf.keras.layers.Lambda(custom_depad)([inputs, x])
    model = tf.keras.Model(inputs,x)

    get_csv_metrics_data_callback = GetCsvMetrics(model, args.csv, args.job_dir, out=args.out)
    get_csv_metrics_data_callback.on_train_end()

