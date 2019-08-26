import tensorflow as tf
import numpy as np

class CycleGAN:   
    def __init__(self, g_AB=None, g_BA=None, d_B=None, d_A=None, patch_gan_hw=2**5, shape = (None, None, 1), verbose=0):
        self.verbose = verbose
        self.shape = shape

        if d_A is None or d_B is None or g_AB is None or g_BA is None:
            raise Exception('d_A, d_B, g_AB, or g_BA cannot be None and must be a `tf.keras.Model`')
        self.d_A = d_A
        self.d_B = d_B
        self.g_AB = g_AB
        self.g_BA = g_BA
        self.patch_gan_hw = patch_gan_hw

    def compile(self, optimizer=tf.keras.optimizers.Adam(0.00002, 0.5), metrics=[], d_loss='mse',
                g_loss = [
                    'mse', 'mse', 
                    'mae', 'mae', 
                    'mae', 'mae'
                ], loss_weights = [
                     1,  1, 
                    10, 10, 
                     1,  1
                ]):
        self.optimizer = optimizer
        self.metrics = metrics
        self.d_loss = d_loss
        self.g_loss = g_loss
        self.loss_weights = loss_weights
        
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
            self.val_batch_placeholder = tf.placeholder(tf.float32)
            self.val_fake_placeholder = tf.placeholder(tf.float32)
            self.output_metrics = {}
            for metric in self.metrics:
                self.output_metrics[metric.__name__] = metric(self.val_batch_placeholder, self.val_fake_placeholder)
        self.metrics_session = tf.Session(graph=self.metrics_graph)
    
    def validate(self, validation_steps):
        metrics_summary = {}
        for metric in self.metrics:
            metrics_summary[metric.__name__] = []
        
        for step in range(validation_steps):
            val_batch = self.sess.run(self.dataset_val_next)
            B_batch = val_batch[1]
            fake_B = self.g_AB.predict(val_batch[0])
                                    
            forward_metrics = self.metrics_session.run(self.output_metrics, feed_dict={
                self.val_batch_placeholder: B_batch,
                self.val_fake_placeholder: fake_B
            })
            
            for key, value in forward_metrics.items():
                if key not in metrics_summary:
                    metrics_summary[key] = []
                metrics_summary[key].append(value)
        
        # average all metrics
        for key, value in metrics_summary.items():
            metrics_summary[key] = np.mean(value)
        return metrics_summary
    
    def fit(self, dataset_a, dataset_b, batch_size=8, steps_per_epoch=10, epochs=3, validation_data=None, validation_steps=10, 
            callbacks=[]):
        
        if not hasattr(self, 'sess'):
            self.dataset_a_next = dataset_a.make_one_shot_iterator().get_next()
            self.dataset_b_next = dataset_b.make_one_shot_iterator().get_next()
            self.dataset_val_next = validation_data.make_one_shot_iterator().get_next()
            
            metrics = ['val_' + metric.__name__ for metric in self.metrics]
            metrics.extend(['d_acc', 'g_loss', 'adv_loss', 'recon_loss', 'id_loss'])
         
            self.sess = tf.Session()
            for callback in callbacks: 
                callback.set_model(self.g_AB)
                callback.set_params({
                    'verbose': self.verbose,
                    'epochs': epochs,
                    'steps': steps_per_epoch,
                    'metrics': metrics
                })
                
        self.log = {
            'size': batch_size
        }
        
        for callback in callbacks: callback.on_train_begin(logs=self.log)
        for epoch in range(epochs):
            for callback in callbacks: callback.on_epoch_begin(epoch, logs=self.log)
            for step in range(steps_per_epoch):
                for callback in callbacks: callback.on_batch_begin(step, logs=self.log)
                
                a_batch = self.sess.run(self.dataset_a_next)
                b_batch = self.sess.run(self.dataset_b_next)
                
                self.patch_gan_size = (a_batch.shape[0],
                                       a_batch.shape[1]//self.patch_gan_hw, 
                                       a_batch.shape[2]//self.patch_gan_hw, 
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
                
                self.log['d_loss'] = d_loss[0]
                self.log['d_acc'] = 100*d_loss[1]
                self.log['g_loss'] = g_loss[0]
                self.log['adv_loss'] = np.mean(g_loss[1:3])
                self.log['recon_loss'] = np.mean(g_loss[3:5])
                self.log['id_loss'] = np.mean(g_loss[5:6])
                                
                for callback in callbacks: callback.on_batch_end(step, logs=self.log)
            
            if validation_data is not None:
                forward_metrics = self.validate(validation_steps)
                for key, value in forward_metrics.items():
                    self.log['val_' + key] = value
            
            for callback in callbacks: callback.on_epoch_end(epoch, logs=self.log)
        for callback in callbacks: callback.on_train_end(logs=self.log)
