import tensorflow as tf
import numpy as np
from keras_subpixel import Subpixel

def PS_2x_upsample(X):
    def _phase_shift(I, r):
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (bsize, a, b, r, r))
        X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
        X = tf.split(1, a, X)  # a, [bsize, b, r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
        X = tf.split(1, b, X)  # b, [bsize, a*r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  #
        bsize, a*r, b*r
        return tf.reshape(X, (bsize, a*r, b*r, 1))
    return _phase_shift(X, 2)

def unet(activation=tf.nn.relu, padding='same', shape=(None, None, 1), 
         filters=[16, 16, 16, 16], bn_filters=16, filter_shape=(3,3), residual=False, pixel_shuffler=False):
    downsample_path = []
    inputs = tf.keras.layers.Input(shape=shape)
    x = inputs
    
    # Downsampling path
    for idx, filter_num in enumerate(filters):
        short_x = tf.keras.layers.Conv2D(filter_num, (1,1), activation=activation, padding=padding)(x)
        x = tf.keras.layers.Conv2D(filter_num, filter_shape, activation=activation, padding=padding)(x)
        x = tf.keras.layers.Conv2D(filter_num, filter_shape, activation=activation, padding=padding)(x)
        x = tf.keras.layers.Add()([x, short_x]) if residual else x
        downsample_path.append(x)
        x = tf.keras.layers.MaxPool2D(padding=padding)(x)
    
    # Bottleneck
    short_x = tf.keras.layers.Conv2D(bn_filters, (1,1), activation=activation, padding=padding)(x)
    x = tf.keras.layers.Conv2D(bn_filters, filter_shape, activation=activation, padding=padding)(x)
    x = tf.keras.layers.Conv2D(bn_filters, filter_shape, activation=activation, padding=padding)(x)
    x = tf.keras.layers.Add()([x, short_x]) if residual else x
    
    downsample_path.reverse()
    filters = list(filters)
    filters.reverse()
    
    # Upsampling path
    for idx, filter_num in enumerate(filters):
        if pixel_shuffler:
            x = Subpixel(filter_num, filter_shape, 2, activation=activation)(x)
        else:
            x = tf.keras.layers.Conv2DTranspose(filter_num, 2, 2, padding=padding)(x)
        x = tf.keras.layers.concatenate([x, downsample_path[idx]])
        
        short_x = tf.keras.layers.Conv2D(bn_filters, (1,1), activation=activation, padding=padding)(x)

        x = tf.keras.layers.Conv2D(filter_num, filter_shape, activation=activation, padding=padding)(x)
        x = tf.keras.layers.Conv2D(filter_num, filter_shape, activation=activation, padding=padding)(x)
        x = tf.keras.layers.Add()([x, short_x]) if residual else x

    outputs = tf.keras.layers.Conv2D(1, 1)(x)
    
    return tf.keras.Model(inputs, outputs)

def patchgan(activation=tf.nn.relu, padding='same', shape=(None, None, 1), 
             filters=[16, 32, 64, 128, 256], filter_shape=(3,3)):
    inputs = tf.keras.Input(shape=shape)
    d = inputs
    for filter_num in filters:
        d = tf.keras.layers.Conv2D(filter_num, filter_shape, activation=activation, padding=padding)(d)
        d = tf.keras.layers.MaxPool2D(padding=padding)(d)
    
    validity = tf.keras.layers.Conv2D(1, 512//(2**(len(filters)+1)), padding=padding)(d)

    return tf.keras.Model(inputs, validity)

class CycleGAN():
    def __init__(self, activation=tf.nn.relu, padding='same', img_shape=(None, None, 1), 
                 g_AB=None, g_BA=None, d_B=None, d_A=None, g_loss = [
                     'mse', 'mse', 
                     'mae', 'mae', 
                     'mae', 'mae'
                 ],
                 generator_params={
                    'filters': [32, 64, 128, 256],
                    'bn_filters': 512,
                    'filter_shape':(3,3)
                 }, discriminator_params = {
                    'filters': [16, 32, 64, 128, 256],
                    'filter_shape': (3,3)
                 }
                ):
        self.img_shape = img_shape
        self.activation = activation
        self.d_activation = activation
        self.padding = padding
        
        self.g_loss = g_loss
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss
        
        self.optimizer = tf.keras.optimizers.Adam(0.00002, 0.5)

        self.generator_params = generator_params
        
        self.discriminator_params = discriminator_params
        
        # Build the discriminator blocks
        if d_A is None:
            self.d_A = self.build_discriminator()
        else:
            self.d_A = d_A
            
        if d_B is None:
            self.d_B = self.build_discriminator()
        else:
            self.d_B = d_B     
        self.d_A.compile(loss='mse',
            optimizer=self.optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=self.optimizer,
            metrics=['accuracy'])
        
        # Build the generator-discriminator block
        self.d_A.trainable = False
        self.d_B.trainable = False
        
        if g_AB is None:
            self.g_AB = self.build_generator()
        else:
            self.g_AB = g_AB
        
        if g_BA is None:
            self.g_BA = self.build_generator()
            self.g_BA = self.build_generator()
        else:
            self.g_BA = g_BA

        img_A = tf.keras.layers.Input(shape=self.img_shape)   # Input images from both domains
        img_B = tf.keras.layers.Input(shape=self.img_shape)
        fake_B = self.g_AB(img_A)                             # Translate images to the other domain
        fake_A = self.g_BA(img_B)
        reconstr_A = self.g_BA(fake_B)                        # Translate images back to original domain
        reconstr_B = self.g_AB(fake_A)
        img_A_id = self.g_BA(img_A)                           # Identity mapping of images
        img_B_id = self.g_AB(img_B)

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)
        
        # Combined model trains generators to fool discriminators
        self.combined = tf.keras.Model(inputs= [img_A, img_B],
                                       outputs=[valid_A, valid_B, 
                                                reconstr_A, reconstr_B, 
                                                img_A_id, img_B_id])
        self.combined.compile(loss=self.g_loss,
                              loss_weights=[1, 1, 
                                            self.lambda_cycle, self.lambda_cycle, 
                                            self.lambda_id, self.lambda_id],
                              optimizer=self.optimizer)
    
    def count_params(self):
        return self.g_AB.count_params()
        
    def build_generator(self):
        "Unet Like Generator"
        return unet(activation=self.activation, padding=self.padding, shape=self.img_shape, 
                    filters=self.generator_params['filters'], 
                    bn_filters=self.generator_params['bn_filters'], 
                    filter_shape=self.generator_params['filter_shape'])

    def build_discriminator(self):
        "Patch GAN Discriminator"
        return patchgan(activation=self.activation, padding=self.padding, shape=self.img_shape, 
                        filters=self.discriminator_params['filters'],
                        filter_shape=self.discriminator_params['filter_shape'])
    
    def fit(self, dataset_a, dataset_b, batch_size=8,
            steps_per_epoch=10, epochs=3, 
            validation_data=None, validation_steps=10, callbacks=[]):
        if not hasattr(self, 'sess'):
            self.dataset_a_next = dataset_a.make_one_shot_iterator().get_next()
            self.dataset_b_next = dataset_b.make_one_shot_iterator().get_next()
            self.dataset_val_next = validation_data.make_one_shot_iterator().get_next()

            self.log = {
                'losses':[],
                'metrics_forward':[],
                'metrics_backward':[]
            }
            self.sess = tf.Session()
            
        for epoch in range(epochs):
            for callback in callbacks: 
                if 'on_epoch_begin' in dir(callback): callback.on_epoch_begin(epoch, logs=self.log)
            for step in range(steps_per_epoch):
                for callback in callbacks: 
                    if 'on_batch_begin' in dir(callback): callback.on_batch_begin(step, logs=self.log)
                   
                a_batch = self.sess.run(self.dataset_a_next)
                b_batch = self.sess.run(self.dataset_b_next)
                self.patch_gan_size = (a_batch.shape[0], a_batch.shape[1]//(2**5), a_batch.shape[2]//(2**5), a_batch.shape[3])
                self.valid = np.ones(self.patch_gan_size)
                self.fake = np.zeros(self.patch_gan_size)                
                
                # Translate images to opposite domain
                fake_B = self.g_AB.predict(a_batch)
                fake_A = self.g_BA.predict(b_batch)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(a_batch, self.valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, self.fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
                dB_loss_real = self.d_B.train_on_batch(b_batch, self.valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, self.fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                g_loss = self.combined.train_on_batch([a_batch, b_batch],
                                                      [self.valid, self.valid, a_batch, b_batch, a_batch, b_batch])
                self.log['losses'].append({
                    'epoch': epoch,
                    'step': step,
                    'd_loss': d_loss[0],
                    'd_acc': 100*d_loss[1],
                    'g_loss': g_loss[0],
                    'adv_loss': np.mean(g_loss[1:3]),
                    'recon_loss': np.mean(g_loss[3:5]),
                    'id_loss': np.mean(g_loss[5:6])
                })
                for callback in callbacks: 
                    if 'on_batch_end' in dir(callback): callback.on_batch_end(step, logs=self.log)
            for callback in callbacks: 
                if 'on_epoch_end' in dir(callback): callback.on_epoch_end(epoch, logs=self.log)

            print('Epoch {}/{}'.format(epoch+1, epochs))
            
            # Create Metrics Graph
            g1 = tf.Graph()
            with g1.as_default():
                A_batch_placeholder = tf.placeholder(tf.float32)
                B_batch_placeholder = tf.placeholder(tf.float32)
                fake_A_placeholder = tf.placeholder(tf.float32)
                fake_B_placeholder = tf.placeholder(tf.float32)
                a_ssim_multiscale = tf.image.ssim_multiscale(A_batch_placeholder, fake_A_placeholder, 1), 
                a_ssim = tf.image.ssim(A_batch_placeholder, fake_A_placeholder, 1),
                a_psnr = tf.image.psnr(A_batch_placeholder, fake_A_placeholder, 1)
                b_ssim_multiscale = tf.image.ssim_multiscale(B_batch_placeholder, fake_B_placeholder, 1), 
                b_ssim = tf.image.ssim(B_batch_placeholder, fake_B_placeholder, 1), 
                b_psnr = tf.image.psnr(B_batch_placeholder, fake_B_placeholder, 1)       
            
            sess_val = tf.Session(graph=g1)   
            val_metrics = {
                'val_ssim_multiscale': [],
                'val_ssim': [],
                'val_psnr': [],
                'val_mse': [],
                'val_mae': []                   
            }

            rev_val_metrics = {
                'val_ssim_multiscale': [],
                'val_ssim': [],
                'val_psnr': [],
                'val_mse': [],
                'val_mae': []                   
            }
            
            for step in range(validation_steps):
                val_batch = self.sess.run(self.dataset_val_next)
                A_batch = val_batch[0]
                B_batch = val_batch[1]
                fake_B = self.g_AB.predict(A_batch)
                fake_A = self.g_BA.predict(B_batch)
                
                rev_val_metrics['val_mse'].append(np.mean(np.power(A_batch - fake_A, 2)))
                val_metrics['val_mse'].append(np.mean(np.power(B_batch - fake_B, 2)))
                rev_val_metrics['val_mae'].append(np.mean(np.abs(A_batch - fake_A)))
                val_metrics['val_mae'].append(np.mean(np.abs(B_batch - fake_B)))
                
                val_ssim_multiscale, val_ssim, val_psnr = sess_val.run([a_ssim_multiscale, a_ssim, a_psnr], feed_dict={
                    A_batch_placeholder: A_batch,
                    fake_A_placeholder: fake_A
                })
                rev_val_metrics['val_ssim_multiscale'].append(val_ssim_multiscale)
                rev_val_metrics['val_ssim'].append(val_ssim)
                rev_val_metrics['val_psnr'].append(val_psnr)
                
                val_ssim_multiscale, val_ssim, val_psnr = sess_val.run([b_ssim_multiscale, b_ssim, b_psnr], feed_dict={
                    B_batch_placeholder: B_batch,
                    fake_B_placeholder: fake_B
                })
                val_metrics['val_ssim_multiscale'].append(val_ssim_multiscale)
                val_metrics['val_ssim'].append(val_ssim)
                val_metrics['val_psnr'].append(val_psnr)
                                
            for key, value in val_metrics.items():
                val_metrics[key] = np.mean(value)

            for key, value in rev_val_metrics.items():
                rev_val_metrics[key] = np.mean(value)
            
            val_metrics['epoch'] =  epoch           
            rev_val_metrics['epoch'] = epoch
            self.log['metrics_forward'].append(val_metrics)
            self.log['metrics_backward'].append(rev_val_metrics)
            
            for callback in callbacks: 
                if 'on_validation_end' in dir(callback): callback.on_validation_end(epoch, logs=self.log)
            
            sess_val.close()
            
        for callback in callbacks: 
            if 'on_train_end' in dir(callback): callback.on_train_end(logs=self.log)
