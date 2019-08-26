import tensorflow as tf
from trainer.config import config
from trainer import utils
from trainer import models
from trainer import callbacks

LOG_DIR = config.job_dir
MODEL_DIR = config.model_dir

# Load Data (Build your custom data loader and replace below)
mimick = utils.MimickDataset(
    clipping=(config.clipping,0), 
    image_dir=config.image_dir,
    shape=(config.in_h, config.in_w)
)
train_dataset, train_count = mimick.get_paired_ultrasound_dataset(
    csv=config.train_csv, 
    batch_size=config.bs
)
train_dataset = train_dataset.map(lambda x,y,z: (x,y))
validation_dataset, val_count = mimick.get_paired_ultrasound_dataset(
    csv=config.validation_csv, 
    batch_size=config.bs
)
validation_dataset = validation_dataset.map(lambda x,y,z: (x,y))
test_dataset, test_count = mimick.get_paired_ultrasound_dataset(
    csv=config.test_csv, batch_size=1
)

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
