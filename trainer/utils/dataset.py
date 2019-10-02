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
import scipy.io as sio
import numpy as np
import polarTransform
import pandas as pd
from functools import partial

DEFAULT_BUCKET_DIR = 'gs://duke-research-us/mimicknet/data/duke-ultrasound-v1'

class MimickDataset():
    def __init__(self, clipping=(-80,0), divisible=16, sc=False, shape=None, image_dir=None, bucket_dir=DEFAULT_BUCKET_DIR):
        self.image_dir = bucket_dir if image_dir is None else image_dir
        self.clipping = clipping
        self.sc = sc
        self.divisible = divisible
        self.shape = shape
        
    def read_mat_op(self, filename, irad, frad, iang, fang):           
        filepath = tf.io.gfile.GFile('{}/{}'.format(self.image_dir, filename.numpy().decode()), 'rb')
        matfile = sio.loadmat(filepath)

        # normalize dtce to [0, 1]
        dtce = matfile['dtce']
        dtce = (dtce - dtce.min())/(dtce.max() - dtce.min())

        # signal detect, clip, and normalize
        iq = np.abs(matfile['iq'])
        if self.clipping is not None:        
            iq = 20*np.log10(iq/iq.max())
            iq = np.clip(iq, self.clipping[0], self.clipping[1])
        else: 
            iq = np.log10(iq)
        iq = (iq-iq.min())/(iq.max() - iq.min())

        if self.sc: # scan convert TODO (this process is heavy so it should be preprocessed)
            iq = scan_convert(iq, irad.numpy(), frad.numpy(), iang.numpy(), fang.numpy())
            dtce = scan_convert(dtce, irad.numpy(), frad.numpy(), iang.numpy(), fang.numpy())

        seed = np.random.randint(0, 2147483647)
        iq, _ = make_shape(iq, shape=self.shape, divisible=self.divisible, seed=seed)
        dtce, _ = make_shape(dtce, shape=self.shape, divisible=self.divisible, seed=seed)  
        return iq.astype('float32'), dtce.astype('float32')
    
    def get_dataset(self, csv):
        count = len(pd.read_csv(tf.io.gfile.GFile(csv, 'rb')))        
        dataset = tf.data.experimental.make_csv_dataset(csv, shuffle=False, batch_size=1).unbatch()
        dataset = dataset.shuffle(count).repeat()
        images  = dataset.map(lambda x: tf.py_function(self.read_mat_op, [x['filename'], 
                                                                          x['initial_radius'], x['final_radius'], 
                                                                          x['initial_angle'], x['final_angle']], [tf.float32, tf.float32]), 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = tf.data.Dataset.zip((images, dataset))
        dataset = dataset.map(lambda x, y: (x[0], x[1], y))
        return dataset, count

    def get_paired_ultrasound_dataset(self, csv='gs://duke-research-us/mimicknet/data/training-v1.csv', batch_size=16):
        dataset, count = self.get_dataset(csv)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size)
        return dataset, count

    def get_unpaired_ultrasound_dataset(self, domain, csv=None, batch_size=16):
        if domain == 'iq':
            csv = 'gs://duke-research-us/mimicknet/data/training_a-v1.csv' if csv is None else csv
            dataset, count = self.get_dataset(csv)
            dataset = dataset.map(lambda iq, dtce, params: (iq, params))
        
        elif domain == 'dtce':
            csv = 'gs://duke-research-us/mimicknet/data/training_b-v1.csv' if csv is None else csv
            dataset, count = self.get_dataset(csv)
            dataset = dataset.map(lambda iq, dtce, params: (dtce, params))
        else:
            raise Exception('domain must be "iq" or "dtce", given {}'.format(domain))
        
        dataset = dataset.batch(batch_size).prefetch(batch_size)
        return dataset, count
    

def make_shape(image, shape=None, divisible=16, seed=0):
    """Will reflection pad or crop to make an image divisible by a number.
    
    If shape is smaller than the original image, it will be cropped randomly
    If shape is larger than the original image, it will be refection padded
    If shape is None, the image's original shape will be minimally padded to be divisible by a number.
    
    Arguments:
        image {np.array} -- np.array that is (height, width, channels)
    
    Keyword Arguments:
        shape {tuple} -- shape of image desired (default: {None})
        seed {number} -- random seed for random cropping (default: {0})
        divisible {number} -- number to be divisible by (default: {16})
    
    Returns:
        np.array, (int, int) -- divisible image no matter the shape, and a tuple of the original size.
    """

    np.random.seed(seed=seed)
    image_height = image.shape[0]
    image_width = image.shape[1]

    shape = shape if shape is not None else image.shape
    height = shape[0] if shape[0] % divisible == 0 else (divisible - shape[0] % divisible) + shape[0]
    width = shape[1] if shape[1] % divisible == 0 else (divisible - shape[1] % divisible) + shape[1]

    # Pad data to batch height and width with reflections, and randomly crop
    if image_height < height:
        remainder = height - image_height
        if remainder % 2 == 0:
            image = np.pad(image, ((int(remainder/2), int(remainder/2)), (0,0)), 'reflect')
        else:
            remainder = remainder - 1
            image = np.pad(image, ((int(remainder/2) + 1, int(remainder/2)), (0,0)), 'reflect')
    elif image_height > height:
        start = np.random.randint(0, image_height - height)
        image = image[start:start+height, :]

    if image_width < width:
        remainder = width - image_width
        if remainder % 2 == 0:
            image = np.pad(image, ((0,0), (int(remainder/2), int(remainder/2))), 'reflect')
        else:
            remainder = remainder - 1
            image = np.pad(image, ((0,0), (int(remainder/2) + 1, int(remainder/2))), 'reflect')
    elif image_width > width:
        start = np.random.randint(0, image_width - width)
        image = image[:, start:start+width]
    image = image[:,:, None]
    
    return image, (image_height, image_width)

def scan_convert(image, irad, frad, iang, fang):
    image, _ = polarTransform.convertToCartesianImage(
        np.transpose(image),
        initialRadius=irad,
        finalRadius=frad,
        initialAngle=iang,
        finalAngle=fang,
        hasColor=False,
        order=1)
    return np.transpose(image[:, int(irad):])
    