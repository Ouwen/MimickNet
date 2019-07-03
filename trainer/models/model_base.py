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
