import tensorflow as tf
import pandas as pd
import scipy.io as sio
import numpy as np
import os
from tensorflow.python.lib.io import file_io
import subprocess
import polarTransform

def ssim_multiscale(y_true, y_pred): return tf.image.ssim_multiscale(y_true, y_pred, 1)
def ssim(y_true, y_pred): return tf.image.ssim(y_true, y_pred, 1)
def psnr(y_true, y_pred): return tf.image.psnr(y_true, y_pred, 1)
def ssim_loss(y_true, y_pred): return 1-tf.image.ssim(y_true, y_pred, 1)
def combined_loss(l_ssim=0.8, l_mae=0.1, l_mse=0.1):
    def _combined_loss(y_true, y_pred):
        return l_ssim*ssim_loss(y_true, y_pred) + l_mae*tf.abs(y_true - y_pred) +  l_mse*tf.square(y_true - y_pred)
    return _combined_loss

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

def get_name(args, model):
    args_dict = args.__dict__
    keys = list(args_dict)
    keys.sort()
    
    name = model + ','
    for key in keys:
        if args_dict[key] == True:
            name += key + ','
        elif args_dict[key] != False and key != 'job_dir':
            name += key + '_' + str(args_dict[key]) + ','
    return name[:-1]

def mat2model(matfile_path, log_compress=True):
    matfile = sio.loadmat(matfile_path)
    iq = abs(matfile['iq'])
    iq = np.log10(iq) if log_compress else iq
    iq = (iq-iq.min())/(iq.max() - iq.min())
    
    dtce = matfile['dtce']
    dtce = (dtce - dtce.min())/(dtce.max() - dtce.min())
    
    shape = iq.shape

    height = iq.shape[0]
    h_pad = 0
    if height % 16 != 0:
        h_pad = 16 - height % 16

    width = iq.shape[1]
    w_pad = 0
    if width % 16 != 0:
        w_pad = 16 - width % 16
    iq = np.pad(iq, [(0, h_pad), (0, w_pad)], 'reflect')
    iq = np.expand_dims(np.expand_dims(iq, axis=0), axis=-1)
    dtce = np.pad(dtce, [(0, h_pad), (0, w_pad)], 'reflect')
    dtce = np.expand_dims(np.expand_dims(dtce, axis=0), axis=-1)
    return iq, dtce, shape

def make_shape(image, shape=None, seed=0):
    np.random.seed(seed=seed)
    image_height = image.shape[0]
    image_width = image.shape[1]

    shape = shape if shape is not None else image.shape
    height = shape[0] if shape[0] % 16 == 0 else (16 - shape[0] % 16) + shape[0]
    width = shape[1] if shape[1] % 16 == 0 else (16 - shape[1] % 16) + shape[1]

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

class MimickDataset():
    def __init__(self, log_compress=True,
                 image_dir=None, bucket_dir='gs://duke-research-us/mimicknet/data/duke-ultrasound-v1'):
        self.log_compress = log_compress
        self.image_dir = image_dir
        self.bucket_dir = bucket_dir

    def read_mat_generator(self, shape=None, sc=False):
        def read_mat_op(filename):           
            def _read_mat(filename, shape=shape):
                filepath = tf.gfile.Open('{}/{}'.format(self.bucket_dir, filename.decode("utf-8")), 'rb')
                if self.image_dir is not None:
                    filepath = '{}/{}'.format(self.image_dir, filename.decode("utf-8"))

                matfile = sio.loadmat(filepath) if not scan_convert else loadmat(filepath)
                dtce = matfile['dtce']
                dtce = (dtce - dtce.min())/(dtce.max() - dtce.min())
                iq = abs(matfile['iq'])
                iq = np.log10(iq) if self.log_compress else iq
                iq = (iq-iq.min())/(iq.max() - iq.min())
                
                if sc:
                    iq = scan_convert(iq, matfile['acq_params'])
                    dtce = scan_convert(dtce, matfile['acq_params'])
                    
                seed = np.random.randint(0, 2147483647)
                iq, _ = make_shape(iq, shape=shape, seed=seed)
                dtce, _ = make_shape(dtce, shape=shape, seed=seed)
                shape = iq.shape
                return iq.astype('float32'), dtce.astype('float32'), shape
            
            output = tf.py_func(_read_mat, [filename], [tf.float32, tf.float32, tf.int64])
            iq = tf.reshape(output[0], shape + (1,))
            dtce = tf.reshape(output[1], shape + (1,))
            return iq, dtce
        return read_mat_op

    def get_dataset(self, csv, shape=None, sc=False):
        filepath = tf.gfile.Open(csv, 'rb')
        filelist = list(pd.read_csv(filepath)['filename'])
        count = len(filelist)
        dataset = tf.data.Dataset.from_tensor_slices(filelist)
        dataset = dataset.shuffle(count).repeat()
        dataset = dataset.map(self.read_mat_generator(shape=shape, sc=sc), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset, count

    def get_paired_ultrasound_dataset(self, csv='data/training-v1.csv', batch_size=16, shape=(512, 64), sc=False):
        dataset, count = self.get_dataset(csv, shape=shape, sc=sc)
        dataset = dataset.batch(batch_size).prefetch(1)
        return dataset, count

    def get_unpaired_ultrasound_dataset(self, domain, csv=None, batch_size=16):
        if domain == 'iq':
            csv = './data/training_a.csv' if csv is None else csv
            dataset, count = self.get_dataset(csv)
            dataset = dataset.map(lambda iq, dtce: iq)
        
        elif domain == 'dtce':
            csv = './data/training_b.csv' if csv is None else csv
            dataset, count = get_dataset(csv)
            dataset = dataset.map(lambda iq, dtce: dtce)
        else:
            raise Exception('domain must be "iq" or "dtce", given {}'.format(domain))
        
        dataset = dataset.batch(batch_size).prefetch(batch_size)
        return dataset, count

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

class GenerateImages(tf.keras.callbacks.Callback):
    def __init__(self, forward, log_dir, interval=1000, log_compress=True,
                 image_dir=None, bucket_dir='gs://duke-research-us/mimicknet/data/duke-ultrasound-v1', 
                 files=[]):
        super()
        self.step_count = 0
        self.interval = interval
        self.forward = forward
        self.writer = tf.summary.FileWriter(log_dir)
        self.files = []
        
        self.graphs = {}
        self.sessions = {}
        self.real_placeholders = {}
        self.fake_placeholders = {}
        self.summaries = {}
        
        # Load files of interest
        for name, filename in files:
            filepath = tf.gfile.Open('{}/{}'.format(bucket_dir, filename), 'rb')
            if image_dir is not None: filepath = '{}/{}'.format(image_dir, filename)
            
            matfile = loadmat(filepath)
            iq = abs(matfile['iq'])
            iq = np.log10(iq) if log_compress else iq
            iq = (iq-iq.min())/(iq.max() - iq.min())
            dtce = matfile['dtce']
            dtce = (dtce - dtce.min())/(dtce.max() - dtce.min())
            
            iq, _ = make_shape(iq)
            dtce, _ = make_shape(dtce)
            iq = np.expand_dims(np.expand_dims(iq, axis=0), axis=-1)
            dtce = np.expand_dims(np.expand_dims(dtce, axis=0), axis=-1)

            acq_params = matfile['acq_params']
            self.files.append((name, iq, dtce, acq_params))
            
            # Create graph of metrics for each image (really annoying tensorboard issue not taking tensor strings)
            self.graphs[name] = tf.Graph()
            with self.graphs[name].as_default():
                self.real_placeholders[name] = tf.placeholder(tf.float32)
                self.fake_placeholders[name]  = tf.placeholder(tf.float32)
                mse = tf.math.reduce_mean(tf.math.square((self.real_placeholders[name] - self.fake_placeholders[name])))
                mae = tf.math.reduce_mean(tf.math.abs((self.real_placeholders[name] - self.fake_placeholders[name])))
                psnr = tf.image.psnr(self.real_placeholders[name], self.fake_placeholders[name], 1)
                ssim = tf.image.ssim(self.real_placeholders[name], self.fake_placeholders[name], 1)
                self.summaries = {
                    'mse': tf.summary.scalar(name, mse, family='mse_val_image'),
                    'mae': tf.summary.scalar(name, mae, family='mae_val_image'),
                    'ssim': tf.summary.scalar(name, ssim[0], family='ssim_val_image'),
                    'psnr': tf.summary.scalar(name, psnr[0], family='psnr_val_image'),
                    'real_image': tf.summary.image('real', self.real_placeholders[name], family=name),
                    'fake_image': tf.summary.image('fake', self.fake_placeholders[name],  family=name),
                    'delta_image': tf.summary.image('delta', tf.abs(self.real_placeholders[name]-self.fake_placeholders[name]),  family=name)
                }
            self.sessions[name] = tf.Session(graph=self.graphs[name])

    def generate_images(self):
        self.step_count += 1

        # Out images on regular interval
        if self.step_count % self.interval == 0:
            # Run the forward model on the images, and write to tensorboard.
            for name, iq, dtce, acq_params in self.files:
                sess = self.sessions[name]
                output = self.forward.predict(iq)
                output = scan_convert(np.squeeze(output), acq_params)
                output = np.expand_dims(np.expand_dims(output, axis=0), axis=-1)
                dtce = scan_convert(np.squeeze(dtce), acq_params)
                dtce = np.expand_dims(np.expand_dims(dtce, axis=0), axis=-1)
                iq = scan_convert(np.squeeze(iq), acq_params)
                iq = np.expand_dims(np.expand_dims(iq, axis=0), axis=-1)

                summary_dict = sess.run(self.summaries, feed_dict = {
                   self.real_placeholders[name]: dtce,
                   self.fake_placeholders[name]: output,
                })
                
                for key, val in summary_dict.items():
                    self.writer.add_summary(val, global_step = self.step_count)
                self.writer.flush()
                
    def on_batch_begin(self, batch, logs={}):
        self.generate_images()
    def on_train_end(self, logs={}):
        self.writer.close()
