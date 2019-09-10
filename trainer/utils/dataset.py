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
        
    def read_mat_op(self, filename):           
        filepath = tf.io.gfile.GFile(filename.numpy(), 'rb')
        matfile = sio.loadmat(filepath) if not self.sc else loadmat(filepath)

        # normalize dtce to [0, 1]
        dtce = matfile['dtce']
        dtce = (dtce - dtce.min())/(dtce.max() - dtce.min())

        # signal detect, clip, and normalize
        iq = np.abs(matfile['iq'])
        if self.clipping is not None:        
            iq = 20*np.log10(iq/iq.max())
            iq = np.clip(iq, self.clipping[0], self.clipping[1])
        elif self.log_compress: 
            iq = np.log10(iq)
        iq = (iq-iq.min())/(iq.max() - iq.min())

        # scan convert TODO (this process is heavy so it should be preprocessed)
        if self.sc:
            iq = scan_convert(iq, matfile['acq_params'])
            dtce = scan_convert(dtce, matfile['acq_params'])

        seed = np.random.randint(0, 2147483647)
        iq, _ = make_shape(iq, shape=self.shape, divisible=self.divisible, seed=seed)
        dtce, _ = make_shape(dtce, shape=self.shape, divisible=self.divisible, seed=seed)  
        return iq.astype('float32'), dtce.astype('float32')
    
    def reshape(iq, dtce, shape):
        output_shape = tf.concat([shape, [1]], axis=0)
        iq = tf.reshape(iq, output_shape)
        dtce = tf.reshape(dtce, output_shape)
        return iq, dtce

    def get_dataset(self, csv):
        filepath = tf.io.gfile.GFile(csv, 'rb')
        filelist = pd.read_csv(filepath)
        filelist['filename'] = filelist['filename'].apply(lambda x: '{}/{}'.format(self.image_dir, x))
        filelist = list(filelist['filename'])
        count = len(filelist)
        
        dataset = tf.data.Dataset.from_tensor_slices(filelist)
        dataset = dataset.shuffle(count).repeat()
        dataset = dataset.map(lambda x: tf.py_function(self.read_mat_op, [x], [tf.float32, tf.float32]), 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset, count

    def get_paired_ultrasound_dataset(self, csv='gs://duke-research-us/mimicknet/data/training-v1.csv', batch_size=16):
        dataset, count = self.get_dataset(csv)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size)
        return dataset, count

    def get_unpaired_ultrasound_dataset(self, domain, csv=None, batch_size=16):
        if domain == 'iq':
            csv = 'gs://duke-research-us/mimicknet/data/training_a.csv' if csv is None else csv
            dataset, count = self.get_dataset(csv)
            dataset = dataset.map(lambda iq, dtce: iq)
        
        elif domain == 'dtce':
            csv = 'gs://duke-research-us/mimicknet/data/training_b.csv' if csv is None else csv
            dataset, count = self.get_dataset(csv)
            dataset = dataset.map(lambda iq, dtce: dtce)
        else:
            raise Exception('domain must be "iq" or "dtce", given {}'.format(domain))
        
        dataset = dataset.batch(batch_size).prefetch(batch_size)
        return dataset, count
    

def loadmat(filename):
    '''
    this function should be called instead of direct sio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, sio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


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

def scan_convert(image, acq_params):
    # Takes an image (r, theta), and acq_params dictionary
    r = acq_params['r']
    apex = acq_params['apex'] if 'apex' in acq_params else acq_params['apex_coordinates'][2]
    theta = acq_params['theta']
    initial_radius = abs((r[0] - apex)/(r[1]-r[0]))
    image, _ = polarTransform.convertToCartesianImage(
        np.transpose(image),
        initialRadius=initial_radius,
        finalRadius=initial_radius+image.shape[0],
        initialAngle=theta[0],
        finalAngle=theta[-1],
        hasColor=False,
        order=3)
    return np.transpose(image[:, int(initial_radius):])
    