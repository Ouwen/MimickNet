import tensorflow as tf
import argparse
from trainer import utils
from trainer import models
from trainer import callbacks
from trainer import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input parser
    parser.add_argument('--bs',       type=int, help='batch size')
    parser.add_argument('--in_h',     type=int, help='image input size height')
    parser.add_argument('--in_w',     type=int, help='image input size width')
    parser.add_argument('--epochs',   type=int, help='number of epochs')
    parser.add_argument('--m',        type=bool, help='manual run or hp tuning')
    parser.add_argument('--train_das_csv', help='csv with das images for training')
    parser.add_argument('--train_clinical_csv', help='csv with clinical images for training')
    parser.add_argument('--validation_csv', help='csv for validation')
    parser.add_argument('--test_csv', help='csv for testing')
    
    # Modeling parser
    parser.add_argument('--clipping', type=float, help='DAS dB clipping')
    parser.add_argument('--kernel_height', type=int, help='height of convolution kernel')
    
    # Cloud ML Params
    parser.add_argument('--job-dir', help='Job directory for Google Cloud ML')
    parser.add_argument('--model_dir', help='Directory for trained models')
    parser.add_argument('--image_dir', help='Local image directory')
    args = parser.parse_args()
    # Merge params
    for key in vars(args):
        if getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))

LOG_DIR = config.job_dir
MODEL_DIR = config.model_dir

# Load Data (Build your custom data loader and replace below)
iq_dataset, iq_count = utils.MimickDataset(
    clipping=(config.clipping,0), 
    image_dir=config.image_dir,
    shape=(config.in_h, config.in_w)
).get_unpaired_ultrasound_dataset(
    domain='iq',
    csv=config.train_das_csv, 
    batch_size=config.bs)
iq_dataset = iq_dataset.map(lambda x,z: x)

dtce_dataset, dtce_count = utils.MimickDataset(
    clipping=(config.clipping,0), 
    image_dir=config.image_dir,
    shape=(config.in_h, config.in_w)
).get_unpaired_ultrasound_dataset(
    domain='dtce',
    csv=config.train_clinical_csv, 
    batch_size=config.bs)
dtce_dataset = dtce_dataset.map(lambda y,z: y)

validation_dataset, val_count = utils.MimickDataset(
    clipping=(config.clipping,0), 
    image_dir=config.image_dir,
    shape=(config.in_h, config.in_w)
).get_paired_ultrasound_dataset(
    csv=config.validation_csv, 
    batch_size=config.bs)
validation_dataset = validation_dataset.map(lambda x,y,z: (x,y))
test_dataset, test_count = utils.MimickDataset(
    clipping=(config.clipping,0)
).get_paired_ultrasound_dataset(csv=config.test_csv, batch_size=1)

if config.is_test: 
    test_count, iq_count, val_count, config.bs = 1,1,1,1

# Select and Compile Model
ModelClass = models.UnetModel
g_AB = ModelClass(shape=(None, None, 1),
                  Activation=tf.keras.layers.LeakyReLU(0.2),
                  filters=[16, 16, 16, 16, 16],
                  filter_shape=(config.kernel_height, 3))()

g_BA = ModelClass(shape=(None, None, 1),
                  Activation=tf.keras.layers.LeakyReLU(0.2),
                  filters=[16, 16, 16, 16, 16],
                  filter_shape=(config.kernel_height, 3))()

d_A = models.PatchDiscriminatorModel(shape=(config.in_h, config.in_w, 1),
                                     Activation=tf.keras.layers.LeakyReLU(0.2),
                                     filters=[32, 64, 128, 256, 512],
                                     filter_shape=(3,3))()

d_B = models.PatchDiscriminatorModel(shape=(config.in_h, config.in_w, 1),
                                     Activation=tf.keras.layers.LeakyReLU(0.2),
                                     filters=[32, 64, 128, 256, 512],
                                     filter_shape=(3,3))()

model = models.CycleGAN(shape = (None, None, 1),
                        g_AB=g_AB,
                        g_BA=g_BA,
                        d_B=d_B,
                        d_A=d_A)

model.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
              d_loss='mse',
              g_loss = [
                 'mse', 'mse',
                 'mae', 'mae',
                 'mae', 'mae'
              ], loss_weights = [
                 1,  1,
                 10, 10,
                 1,  1
              ],
              metrics=[utils.ssim, utils.psnr, utils.mae, utils.mse])

# Generate Callbacks
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, write_graph=True, update_freq='epoch')
start_tensorboard = callbacks.StartTensorBoard(LOG_DIR)

prog_bar = tf.keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None)
log_code = callbacks.LogCode(LOG_DIR, './trainer')
copy_keras = callbacks.CopyKerasModel(MODEL_DIR, LOG_DIR)

saving = callbacks.MultiModelCheckpoint(MODEL_DIR + '/model.{epoch:02d}-{val_ssim:.10f}.hdf5',
                                        monitor='val_ssim', verbose=1, freq='epoch', mode='max', save_best_only=True,
                                        multi_models=[('g_AB', g_AB), ('g_BA', g_BA), ('d_A', d_A), ('d_B', d_B)])

reduce_lr = callbacks.MultiReduceLROnPlateau(training_models=[model.d_A, model.d_B, model.combined], 
                                             monitor='val_ssim', mode='max', factor=0.5, patience=3, min_lr=0.000002)
early_stopping = callbacks.MultiEarlyStopping(multi_models=[g_AB, g_BA, d_A, d_B], full_model=model,
                                              monitor='val_ssim', mode='max', patience=1, 
                                              restore_best_weights=True, verbose=1)

image_gen = callbacks.GenerateImages(g_AB, validation_dataset, LOG_DIR, interval=int(iq_count/config.bs))
get_csv_metrics = callbacks.GetCsvMetrics(g_AB, test_dataset, LOG_DIR, count=test_count)

# Fit the model
model.fit(iq_dataset, dtce_dataset,
          steps_per_epoch=int(iq_count/config.bs),
          epochs=config.epochs,
          validation_data=validation_dataset,
          validation_steps=int(val_count/config.bs),
          callbacks=[log_code, reduce_lr, tensorboard, prog_bar, image_gen, saving, 
                     copy_keras, start_tensorboard, get_csv_metrics, early_stopping])
