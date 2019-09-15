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
    parser.add_argument('--train_csv', help='csv for paired training')
    parser.add_argument('--validation_csv', help='csv for validation')
    parser.add_argument('--test_csv', help='csv for testing')
    
    # Cloud ML Params
    parser.add_argument('--job-dir', help='Job directory for Google Cloud ML')
    parser.add_argument('--model_dir', help='Directory for trained models')
    parser.add_argument('--image_dir', help='Local image directory')
    args = parser.parse_args()
    # Merge params
    for key in vars(args):
        if getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))

print(config.__dict__)

LOG_DIR = config.job_dir
MODEL_DIR = config.model_dir

train_dataset, train_count = utils.MimickDataset(
    clipping=(config.clipping,0), 
    image_dir=config.image_dir,
    shape=(config.in_h, config.in_w)
).get_paired_ultrasound_dataset(
    csv=config.train_csv, 
    batch_size=config.bs)
train_dataset = train_dataset.map(lambda x,y,z: (x,y))

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
    test_count, train_count, val_count, config.bs = 1,1,1,1
    
ModelClass = models.UnetModel
model = ModelClass(shape=(None, None, 1),
                   Activation=tf.keras.layers.ReLU(),
                   filters=[16, 16, 16, 16, 16],
                   filter_shape=(config.kernel_height, 3))()
    
model.compile(optimizer=tf.keras.optimizers.Adam(0.002), loss='mae', metrics=[utils.mae, utils.mse, utils.ssim, utils.psnr])

# Generate Callbacks
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, write_graph=True, update_freq='epoch')
copy_keras = callbacks.CopyKerasModel(MODEL_DIR, LOG_DIR)
saving = tf.keras.callbacks.ModelCheckpoint(MODEL_DIR + '/model.{epoch:02d}-{val_ssim:.10f}.hdf5', 
                                            monitor='val_ssim', verbose=1, save_freq='epoch', mode='max', save_best_only=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.002*.001)
log_code = callbacks.LogCode(LOG_DIR, './trainer')
terminate = tf.keras.callbacks.TerminateOnNaN()
image_gen = callbacks.GenerateImages(model, validation_dataset, LOG_DIR, interval=int(train_count/config.bs))
get_csv_metrics = callbacks.GetCsvMetrics(model, test_dataset, LOG_DIR, count=test_count)

# Fit the model
model.fit(train_dataset,
          steps_per_epoch=int(train_count/config.bs),
          epochs=config.epochs,
          validation_data=validation_dataset,
          validation_steps=int(val_count/config.bs),
          verbose=1,
          callbacks=[log_code, terminate, tensorboard, saving, reduce_lr, copy_keras, image_gen, get_csv_metrics])
