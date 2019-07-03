import tensorflow as tf
from trainer.models.model_base import ModelBase

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
