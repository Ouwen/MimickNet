import tensorflow as tf
import argparse
import numpy as np

from trainer.utils.dataset import loadmat, make_shape, scan_convert
from trainer.utils import custom_ssim

class GenerateImages(tf.keras.callbacks.Callback):
    def __init__(self, forward, log_dir, interval=1000, log_compress=True, clipping=True,
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
        self.log_compress = log_compress
        self.clipping = clipping
        
        # Load files of interest
        for name, filename in files:
            filepath = tf.gfile.Open('{}/{}'.format(bucket_dir, filename), 'rb')
            if image_dir is not None: filepath = '{}/{}'.format(image_dir, filename)
            
            matfile = loadmat(filepath)
            iq = abs(matfile['iq'])
            if self.clipping:
                iq = 20*np.log10(iq/iq.max())
                iq = np.clip(iq, -80, 0)
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
            self.files.append((name, iq, dtce, acq_params))
            
            # Create graph of metrics for each image (really annoying tensorboard issue not taking tensor strings)
            self.graphs[name] = tf.Graph()
            with self.graphs[name].as_default():
                self.real_placeholders[name] = tf.placeholder(tf.float32)
                self.fake_placeholders[name]  = tf.placeholder(tf.float32)
                mse = tf.math.reduce_mean(tf.math.square((self.real_placeholders[name] - self.fake_placeholders[name])))
                mae = tf.math.reduce_mean(tf.math.abs((self.real_placeholders[name] - self.fake_placeholders[name])))
                psnr = tf.image.psnr(self.real_placeholders[name], self.fake_placeholders[name], 1)
                val, cs, l = custom_ssim.ssim(self.real_placeholders[name], self.fake_placeholders[name], 1)
                self.summaries[name] = {
                    'mse': tf.summary.scalar(name, mse, family='mse_val_image'),
                    'mae': tf.summary.scalar(name, mae, family='mae_val_image'),
                    'ssim': tf.summary.scalar(name, val[0], family='ssim_val_image'),
                    'contrast_structure': tf.summary.scalar(name, cs[0], family='contrast_structure_val_image'),
                    'luminance': tf.summary.scalar(name, l[0], family='luminance_val_image'),
                    'psnr': tf.summary.scalar(name, psnr[0], family='psnr_val_image'),
                    'real_image': tf.summary.image('real', self.real_placeholders[name], family=name),
                    'fake_image': tf.summary.image('fake', self.fake_placeholders[name],  family=name),
                    'delta_image': tf.summary.image('delta', tf.abs(self.real_placeholders[name]-self.fake_placeholders[name]), family=name)
                }
            self.sessions[name] = tf.Session(graph=self.graphs[name])

    def generate_images(self):
        # Run the forward model on the images, and write to tensorboard.
        for name, iq, dtce, acq_params in self.files:
            sess = self.sessions[name]
            output = self.forward.predict(iq)
                
            output = scan_convert(np.squeeze(output), acq_params)
            output = np.clip(output, 0, 1)
            output = np.expand_dims(np.expand_dims(output, axis=0), axis=-1)
                
            dtce = scan_convert(np.squeeze(dtce), acq_params)
            dtce = np.clip(dtce, 0, 1)
            dtce = np.expand_dims(np.expand_dims(dtce, axis=0), axis=-1)
                
            iq = scan_convert(np.squeeze(iq), acq_params)
            iq = np.clip(iq, 0, 1)
            iq = np.expand_dims(np.expand_dims(iq, axis=0), axis=-1)
                
            summary_dict = sess.run(self.summaries[name], feed_dict = {
                self.real_placeholders[name]: dtce,
                self.fake_placeholders[name]: output,
            })
                
            for key, val in summary_dict.items():
                self.writer.add_summary(val, global_step = self.step_count)
            self.writer.flush()
                
    def on_batch_begin(self, batch, logs={}):
        self.step_count += 1
        if self.step_count % self.interval == 0:
            self.generate_images()
            
    def on_train_end(self, logs={}):
        self.generate_images()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--forward', default=None, help='Forward .hdf5 model filepath')
    parser.add_argument('--log_dir', default='.', help='Metrics ouptut filepath')
    parser.add_argument('--lg_c',     default= True,  type=bool, help='Log compress the raw IQ data')
    parser.add_argument('--clip',     default= True,  type=bool, help='Clip to -80 of raw beamformed data')
    parser.add_argument('--bucket_dir', default='gs://duke-research-us/mimicknet/data/duke-ultrasound-v1', help='Job directory for Google Cloud ML')
    parser.add_argument('--image_dir', default=None)
    
    args = parser.parse_args()
    
    if args.forward is None:
        raise ValueError("{} is not valid directory".format(args.forward))
    
    def randomfunc(x,y): return tf.image.ssim(x,y, max_val=1)
    model = tf.keras.models.load_model(args.forward, custom_objects={'ssim_loss': randomfunc,
                                                                     'ssim': randomfunc,
                                                                     'ssim_multiscale': randomfunc,
                                                                     'psnr':randomfunc})
    
    generate_images_callback = GenerateImages(model, args.log_dir, 
                                              log_compress = args.lg_c, 
                                              clipping = args.clip, 
                                              bucket_dir=args.bucket_dir, 
                                              image_dir=args.image_dir, 
                                              files=[
                                                        ('fetal', 'rfd_fetal_ch.uri_SpV5192_VpF1362_FpA6_20121101150345_1.mat'),
                                                        ('fetal2', 'rfd_fetal_ch.uri_SpV6232_VpF908_FpA9_20121031150931_1.mat'),                        
                                                        ('liver', 'rfd_liver_highmi.uri_SpV5192_VpF512_FpA7_20161216093626_1.mat'),
                                                        ('phantom', 'verasonics.20180206194115_channelData_part11_0.mat'),
                                                        ('vera_bad', 'verasonics.20170830145820_channelData_part9_1.mat'),
                                                        ('sc_bad', 'sc_fetal_ch.20160909160351_sum_10.mat'),
                                                        ('rfd_bad', 'rfd_liver_highmi.uri_SpV10388_VpF168_FpA8_20160901073342_2.mat')
                                                    ])
    generate_images_callback.generate_images()
