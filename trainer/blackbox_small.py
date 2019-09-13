import tensorflow as tf
import argparse
import sys
import time
from trainer import utils
from trainer import models
from trainer import callbacks

def main(argv):
    args = parser.parse_args()
    LOG_DIR = args.job_dir # Directory to save logs
    MODEL_DIR = './trained_models' # Directory to output models
    
    # Load Data (Build your custom data loader and replace below)
    iq_dataset, iq_count = utils.MimickDataset(
        clipping=(-80,0), 
        image_dir=args.image_dir,
        shape=(args.in_h, args.in_w)
    ).get_unpaired_ultrasound_dataset(
        domain='iq',
        csv='./data/training_a-v1.csv', 
        batch_size=args.bs)
    iq_dataset = iq_dataset.map(lambda x,z: x)

    dtce_dataset, dtce_count = utils.MimickDataset(
        clipping=(-80,0), 
        image_dir=args.image_dir,
        shape=(args.in_h, args.in_w)
    ).get_unpaired_ultrasound_dataset(
        domain='dtce',
        csv='./data/training_b-v1.csv', 
        batch_size=args.bs)
    dtce_dataset = dtce_dataset.map(lambda y,z: y)
    validation_dataset, val_count = utils.MimickDataset(
        clipping=(-80,0), 
        image_dir=args.image_dir,
        shape=(args.in_h, args.in_w)
    ).get_paired_ultrasound_dataset(
        csv='./data/testing-v1.csv', 
        batch_size=args.bs)
    validation_dataset = validation_dataset.map(lambda x,y,z: (x,y))
    test_dataset, test_count = utils.MimickDataset().get_paired_ultrasound_dataset(csv='./data/testing-v1.csv', batch_size=1)


    # Select and Compile Model
    ModelClass = models.UnetModel
    g_AB = ModelClass(shape=(None, None, 1),
                      Activation=tf.keras.layers.LeakyReLU(0.2),
                      filters=[16, 16, 16, 16, 16],
                      filter_shape=(7, 3))()

    g_BA = ModelClass(shape=(None, None, 1),
                      Activation=tf.keras.layers.LeakyReLU(0.2),
                      filters=[16, 16, 16, 16, 16],
                      filter_shape=(7, 3))()

    d_A = models.PatchDiscriminatorModel(shape=(args.in_h, args.in_w, 1),
                                         Activation=tf.keras.layers.LeakyReLU(0.2),
                                         filters=[32, 64, 128, 256, 512],
                                         filter_shape=(3,3))()

    d_B = models.PatchDiscriminatorModel(shape=(args.in_h, args.in_w, 1),
                                         Activation=tf.keras.layers.LeakyReLU(0.2),
                                         filters=[32, 64, 128, 256, 512],
                                         filter_shape=(3,3))()
    
    g_AB.load_weights('{}/{}-g_AB.hdf5'.format(MODEL_DIR, 'model.02-0.8907387257'))
    g_BA.load_weights('{}/{}-g_BA.hdf5'.format(MODEL_DIR, 'model.02-0.8907387257'))
    d_A.load_weights('{}/{}-d_A.hdf5'.format(MODEL_DIR, 'model.02-0.8907387257'))
    d_B.load_weights('{}/{}-d_B.hdf5'.format(MODEL_DIR, 'model.02-0.8907387257'))

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
                                                 monitor='val_ssim', mode='max', factor=1e-1, patience=1)
    early_stopping = callbacks.MultiEarlyStopping(multi_models=[g_AB, g_BA, d_A, d_B], full_model=model,
                                                  monitor='val_ssim', mode='max', patience=1, 
                                                  restore_best_weights=True, verbose=1)
    
    image_gen = callbacks.GenerateImages(g_AB, validation_dataset, LOG_DIR, interval=int(iq_count/args.bs))
    get_csv_metrics = callbacks.GetCsvMetrics(g_AB, test_dataset, LOG_DIR, count=test_count)
    
    # Fit the model
    model.fit(iq_dataset, dtce_dataset,
              steps_per_epoch=int(iq_count/args.bs),
              epochs=args.epochs,
              validation_data=validation_dataset,
              validation_steps=int(val_count/args.bs),
              callbacks=[log_code, reduce_lr, tensorboard, prog_bar, image_gen, saving, 
                         copy_keras, start_tensorboard, get_csv_metrics, early_stopping])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input parser
    parser.add_argument('--bs',       default= 8, type=int, help='batch size')
    parser.add_argument('--in_h',     default= 512, type=int, help='image input size height')
    parser.add_argument('--in_w',     default= 512, type=int, help='image input size width')
    parser.add_argument('--epochs',   default= 100, type=int, help='number of epochs')
    parser.add_argument('--m',        default= True, type=bool, help='manual run or hp tuning')

    # Cloud ML Params
    parser.add_argument('--job-dir', default='gs://duke-research-us/mimicknet/revision_experiments/tmp/{}'.format(str(time.time())), 
                        help='Job directory for Google Cloud ML')
    parser.add_argument('--image_dir', default='./data/duke-ultrasound-v1', help='Local image directory')
    
    main(sys.argv)
