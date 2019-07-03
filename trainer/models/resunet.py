import tensorflow as tf
from trainer.models.model_base import ModelBase

class ResUnetModel(ModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def resblock(self, x, num_filters, first=False):
        if first:
            shortcut = x
        else:
            shortcut = tf.keras.layers.Conv2D(num_filters, kernel_size=(1, 1))(x)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
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