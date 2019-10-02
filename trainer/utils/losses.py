"""
Copyright Ouwen Huang 2019 

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf

def ssim_multiscale(y_true, y_pred): return tf.image.ssim_multiscale(y_true, y_pred, 1)
def ssim(y_true, y_pred): return tf.image.ssim(y_true, y_pred, 1)
def psnr(y_true, y_pred): return tf.image.psnr(y_true, y_pred, 1)
def ssim_loss(y_true, y_pred): return 1-tf.image.ssim(y_true, y_pred, 1)
def combined_loss(l_ssim=0.8, l_mae=0.1, l_mse=0.1):
    def _combined_loss(y_true, y_pred):
        return l_ssim*ssim_loss(y_true, y_pred) + l_mae*tf.abs(y_true - y_pred) +  l_mse*tf.square(y_true - y_pred)
    return _combined_loss

def mse(x, y): return tf.reshape(tf.math.reduce_mean(tf.math.square((x - y))), [-1])
def mae(x, y): return tf.reshape(tf.math.reduce_mean(tf.math.abs((x - y))), [-1])
