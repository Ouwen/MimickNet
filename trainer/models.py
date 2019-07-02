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
        if l1_regularizer is 0 and l2_regularizer is 0:
            self.kernel_regularizer = None
        else:
            self.kernel_regularizer = tf.keras.regularizers.l1_l2(l1=l1_regularizer, l2=l2_regularizer)
    
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

class PatchDiscriminatorModel(ModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __call__(self):
        inputs = tf.keras.Input(shape=self.shape)
        x = inputs
        for filter_num in self.filters:
            x = self.Conv2D(filter_num)(x)
            x = self.Activation(x)
            x = tf.keras.layers.MaxPool2D(padding=self.padding)(x)
        validity = tf.keras.layers.Conv2D(1, (self.shape[0]//(2**(len(self.filters)+1)), 
                                              self.shape[1]//(2**(len(self.filters)+1))), padding=self.padding)(x)
        return tf.keras.Model(inputs, validity)
