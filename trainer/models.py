import tensorflow as tf

class ModelBase:
    def __init__(self, Activation=tf.keras.layers.ReLU, 
                 padding='same', 
                 shape=(None, None, 1),
                 filters=[16, 16, 16, 16, 16], filter_shape=(3,3), 
                 pixel_shuffler=False,
                 dropout_rate=0, l1_regularizer=0, l2_regularizer=0):
    
        self.Activation         = Activation
        self.padding            ='same'
        self.filter_shape       = filter_shape
        self.shape              = shape
        self.filters            = filters
        self.pixel_shuffler     = pixel_shuffler
        self.dropout_rate       = dropout_rate
        self.kernel_regularizer = tf.keras.regularizers.l1_l2(l1=l1_regularizer, l2=l2_regularizer) if l1_regularizer != 0 and l2_regularizer !=0 else None    
    
    def Conv2D(self, filters):
        return tf.keras.layers.Conv2D(filters, self.filter_shape, padding=self.padding, kernel_regularizer=self.kernel_regularizer)
    
    def Upsample(self, filters, ratio=2):
        if self.pixel_shuffler:
            return tf.keras.Sequential([tf.keras.layers.Lambda(lambda x: tf.depth_to_space(x, ratio)),
                                        self.Conv2D(filters)])
        else:
            return tf.keras.layers.Conv2DTranspose(filters, ratio, ratio, padding=self.padding)
    def Dropout(self):
        def _dropout(x):
            return tf.keras.layers.AlphaDropout(self.dropout_rate)(x) if self.dropout_rate > 0 else x
        return _dropout
    
class UnetModel(ModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
            
    def __call__(self):
        downsample_path = []
        inputs = tf.keras.layers.Input(shape=self.shape)
        x = inputs
        
        for idx, filter_num in enumerate(self.filters):
            x = self.Conv2D(filter_num)(x)
            x = self.Activation(x)
            x = self.Dropout()(x)
            x = self.Conv2D(filter_num)(x)
            x = self.Activation(x)
            x = self.Dropout()(x)
            if idx != len(self.filters)-1:
                downsample_path.append(x)
                x = tf.keras.layers.MaxPool2D(padding=self.padding)(x)

        downsample_path.reverse()
        reverse_filters = list(self.filters[:-1])
        reverse_filters.reverse()

        # Upsampling path
        for idx, filter_num in enumerate(reverse_filters):
            x = self.Upsample(filter_num)(x)
            x = tf.keras.layers.concatenate([x, downsample_path[idx]])
            x = self.Conv2D(filter_num)(x)
            x = self.Activation(x)
            x = self.Dropout()(x)
            x = self.Conv2D(filter_num)(x)
            x = self.Activation(x)
            x = self.Dropout()(x)

        x = tf.keras.layers.Conv2D(1, 1)(x)

        return tf.keras.Model(inputs, x)

class ResUnetModel(ModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def resblock(self, x, num_filters, first=False):
        shortcut = tf.keras.layers.Conv2D(num_filters, kernel_size=(1, 1))(x)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        if not first:
            x = tf.keras.layers.BatchNormalization()(x)
            x = self.Activation(x)
        res_path = self.Conv2D(num_filters)(x)
        res_path = tf.keras.layers.BatchNormalization()(res_path)
        res_path = self.Activation(res_path)
        res_path = self.Conv2D(num_filters)(res_path)
        return tf.keras.layers.Add()([shortcut, res_path])
    
    def __call__(self):
        downsample_path = []
        inputs = tf.keras.layers.Input(shape=self.shape)
        x = inputs

        for idx, filter_num in enumerate(self.filters):
            x = self.resblock(x, filter_num, idx==0)
            if idx != len(self.filters)-1:
                downsample_path.append(x)
                x = tf.keras.layers.MaxPool2D(padding=self.padding)(x)

        downsample_path.reverse()
        reverse_filters = list(self.filters[:-1])
        reverse_filters.reverse()

        # Upsampling path
        for idx, filter_num in enumerate(reverse_filters):
            x = self.Upsample(filter_num)(x)
            x = tf.keras.layers.concatenate([x, downsample_path[idx]])        
            x = self.resblock(x, filter_num)
        x = tf.keras.layers.Conv2D(1, 1)(x)

        return tf.keras.Model(inputs, x)

    
def patchgan(activation=tf.nn.relu, padding='same', shape=(None, None, 1), 
             filters=[16, 32, 64, 128, 256], filter_shape=(3,3)):
    inputs = tf.keras.Input(shape=shape)
    d = inputs
    for filter_num in filters:
        d = tf.keras.layers.Conv2D(filter_num, filter_shape, activation=activation, padding=padding)(d)
        d = tf.keras.layers.MaxPool2D(padding=padding)(d)
    
    validity = tf.keras.layers.Conv2D(1, 512//(2**(len(filters)+1)), padding=padding)(d)

    return tf.keras.Model(inputs, validity)

class CycleGAN(ModelBase):   
    def __init__(self, g_AB=None, g_BA=None, d_B=None, d_A=None, 
                 d_loss='mse',
                 g_loss = [
                     'mse', 'mse', 
                     'mae', 'mae', 
                     'mae', 'mae'
                 ], loss_weights = [
                     1,  1, 
                    10, 10, 
                     1,  1
                 ], generator_params={
                    'filters': [32, 64, 128, 256, 512],
                    'filter_shape':(3,3)
                 }, discriminator_params = {
                    'filters': [16, 32, 64, 128, 256],
                    'filter_shape': (3,3)
                 }, **kwargs):
        super().__init__(**kwargs)
        
        self.d_loss = d_loss
        self.g_loss = g_loss
        self.loss_weight = loss_weight
        
        self.generator_params = generator_params
        self.discriminator_params = discriminator_params
        
        # Build the discriminator blocks
        self.d_A = self.build_discriminator() if d_A is None else d_A
        self.d_B = self.build_discriminator() if d_B is None else d_B
        
        # Build the generator blocks
        self.g_AB = self.build_generator() if g_AB is None else g_AB
        self.g_BA = self.build_generator() if g_BA is None else g_BA
        
    def compile(self, optimizer=tf.keras.optimizers.Adam(0.00002, 0.5), metrics=[], log_dir=None):
        self.optimizer = optimizer
        self.d_A.compile(loss=self.d_loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.d_B.compile(loss=self.d_loss, optimizer=self.optimizer, metrics=['accuracy'])
        
        # Build the generator block
        self.d_A.trainable = False
        self.d_B.trainable = False
        
        img_A = tf.keras.layers.Input(shape=self.shape)   # Input images from both domains
        img_B = tf.keras.layers.Input(shape=self.shape)
        fake_B = self.g_AB(img_A)                         # Translate images to the other domain
        fake_A = self.g_BA(img_B)
        reconstr_A = self.g_BA(fake_B)                    # Translate images back to original domain
        reconstr_B = self.g_AB(fake_A)
        img_A_id = self.g_BA(img_A)                       # Identity mapping of images
        img_B_id = self.g_AB(img_B)

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)
        
        # Combined model trains generators to fool discriminators
        self.combined = tf.keras.Model(inputs  = [img_A, img_B],
                                       outputs = [valid_A, valid_B, 
                                                  reconstr_A, reconstr_B, 
                                                  img_A_id, img_B_id])
        self.combined.compile(loss=self.g_loss,
                              loss_weights=self.loss_weights,
                              optimizer=self.optimizer)
        
        self.metrics_graph = tf.Graph()
        with self.metrics_graph.as_default():
            A_batch_placeholder = tf.placeholder(tf.float32)
            B_batch_placeholder = tf.placeholder(tf.float32)
            fake_A_placeholder = tf.placeholder(tf.float32)
            fake_B_placeholder = tf.placeholder(tf.float32)
            self.output_metrics = [metric(A_batch_placeholder, fake_A_placeholder) for metric in metrics]

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
    
    def fit(self, dataset_a, dataset_b, batch_size=8, steps_per_epoch=10, epochs=3, validation_data=None, validation_steps=10, 
            callbacks=[]):
        
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
        
        for callback in callbacks: callback.on_train_begin(logs=self.log)
        for epoch in range(epochs):
            for callback in callbacks: callback.on_epoch_begin(epoch, logs=self.log)
            for step in range(steps_per_epoch):
                for callback in callbacks: callback.on_batch_begin(step, logs=self.log)
                
                a_batch = self.sess.run(self.dataset_a_next)
                b_batch = self.sess.run(self.dataset_b_next)
                
                self.patch_gan_size = (a_batch.shape[0], 
                                       a_batch.shape[1]//(2**len(self.discriminator_params['filters'])), 
                                       a_batch.shape[2]//(2**len(self.discriminator_params['filters'])), 
                                       a_batch.shape[3])
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
                for callback in callbacks: callback.on_batch_end(step, logs=self.log)
            for callback in callbacks: callback.on_epoch_end(epoch, logs=self.log)

            print('Epoch {}/{}'.format(epoch+1, epochs))
            
            sess_val = tf.Session(graph=g1)
            
            
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
