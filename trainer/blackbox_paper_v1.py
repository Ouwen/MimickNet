import tensorflow as tf
from trainer import utils
from trainer import models
from trainer import callbacks

LOG_DIR = './logs'
MODEL_DIR = '.'

# Load Data (Build your custom data loader and replace below)
mimick = utils.MimickDataset(clipping=None, shape=(512, 512), image_dir=None)
iq_dataset, iq_count = mimick.get_unpaired_ultrasound_dataset(
    domain='iq',
    csv='./data/training_a-v1.csv', 
    batch_size=8)
iq_dataset = iq_dataset.map(lambda x,z: x)

dtce_dataset, dtce_count = mimick.get_unpaired_ultrasound_dataset(
    domain='dtce',
    csv='./data/training_b-v1.csv', 
    batch_size=8)
dtce_dataset = dtce_dataset.map(lambda x,z: x)

test_dataset, val_count = mimick.get_paired_ultrasound_dataset(
    csv='./data/testing-v1.csv', 
    batch_size=8)
test_dataset = test_dataset.map(lambda x,y,z: (x,y))

# Load Model
ModelClass = models.UnetModel
g_AB = ModelClass(shape=(None, None, 1),
                  Activation=tf.keras.layers.ReLU(),
                  filters=[16, 16, 16, 16, 16],
                  filter_shape=(7,3))()
g_BA = ModelClass(shape=(None, None, 1),
                  Activation=tf.keras.layers.ReLU(),
                  filters=[16, 16, 16, 16, 16],
                  filter_shape=(7,3))()
d_A = models.PatchDiscriminatorModel(shape=(512, 512, 1),
                                     Activation=tf.keras.layers.ReLU(),
                                     filters=[32, 64, 128, 256, 512],
                                     filter_shape=(3,3))()
d_B = models.PatchDiscriminatorModel(shape=(512, 512, 1),
                                     Activation=tf.keras.layers.ReLU(),
                                     filters=[32, 64, 128, 256, 512],
                                     filter_shape=(3,3))()

model = models.CycleGAN(shape = (None, None, 1),
                        g_AB=g_AB,
                        g_BA=g_BA,
                        d_B=d_B,
                        d_A=d_A)

# Compile Model and Set Lambda Hyperparams
model.compile(optimizer=tf.keras.optimizers.Adam(0.00002, 0.5),
              d_loss='mse',
              g_loss = [
                 'mse', 'mse', 
                 'mse', 'mse', 
                 'mse', 'mse'
              ], loss_weights = [
                 1,  1,
                 10, 10,
                 1,  1
              ],
              metrics=[utils.ssim, utils.psnr, utils.mse, utils.mae])

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, write_graph=True, update_freq='epoch')
start_tensorboard = callbacks.StartTensorBoard(LOG_DIR)
prog_bar = tf.keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None)

# Fit the model
model.fit(iq_dataset, dtce_dataset,
          steps_per_epoch=int(iq_count/8),
          epochs=50,
          validation_data=test_dataset,
          validation_steps=int(val_count/8),
          callbacks=[tensorboard, start_tensorboard, prog_bar])
