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
    MODEL_DIR = '.' # Directory to output models
    
    # Load Data (Build your custom data loader and replace below)
    mimick_dataset = utils.MimickDataset(log_compress=True, clipping=True, image_dir=args.image_dir)
    iq_dataset, iq_count = mimick_dataset.get_unpaired_ultrasound_dataset(
        domain='iq',
        csv='./data/training_a.csv', 
        batch_size=args.bs, 
        shape=(args.in_h, args.in_w),
        sc=False)
    dtce_dataset, dtce_count = mimick_dataset.get_unpaired_ultrasound_dataset(
        domain='dtce',
        csv='./data/training_b.csv', 
        batch_size=args.bs, 
        shape=(args.in_h, args.in_w),
        sc=False)
    test_dataset, val_count = mimick_dataset.get_paired_ultrasound_dataset(
        csv='./data/testing-v1.csv', 
        batch_size=args.bs,
        shape=(args.in_h, args.in_w),
        sc=False)

    # Select and Compile Model
    ModelClass = models.UnetModel
    g_AB = ModelClass(shape=(None, None, 1),
                      Activation=tf.keras.layers.LeakyReLU(0.2),
                      filters=[8, 16, 32, 64, 128],
                      filter_shape=(3, 3))()

    g_BA = ModelClass(shape=(None, None, 1),
                      Activation=tf.keras.layers.LeakyReLU(0.2),
                      filters=[8, 16, 32, 64, 128],
                      filter_shape=(3, 3))()

    d_A = models.PatchDiscriminatorModel(shape=(args.in_h, args.in_w, 1),
                                         Activation=tf.keras.layers.LeakyReLU(0.2),
                                         filters=[16, 32, 64, 128, 256],
                                         filter_shape=(3,3))()
    

    d_B = models.PatchDiscriminatorModel(shape=(args.in_h, args.in_w, 1),
                                         Activation=tf.keras.layers.LeakyReLU(0.2),
                                         filters=[16, 32, 64, 128, 256],
                                         filter_shape=(3,3))()
    
    model = models.CycleGAN(verbose = 1,
                            shape = (None, None, 1),
                            g_AB=g_AB,
                            g_BA=g_BA,
                            d_B=d_B,
                            d_A=d_A,
                            patch_gan_hw=2**5)
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
                  metrics=[utils.ssim, utils.psnr])
    
    # Generate Callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, write_graph=True, update_freq='epoch')
    prog_bar = tf.keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None)
    log_code = callbacks.LogCode(LOG_DIR, './trainer')
    copy_keras = callbacks.CopyKerasModel(MODEL_DIR, LOG_DIR)
    get_csv = callbacks.GetCsvMetrics(g_AB, LOG_DIR)    
    save_multi_model = callbacks.SaveMultiModel([('g_AB', g_AB), ('g_BA', g_BA), ('d_A', d_A), ('d_B', d_B)], MODEL_DIR)
    saving = tf.keras.callbacks.ModelCheckpoint(MODEL_DIR + '/model.{epoch:02d}-{val_ssim:.10f}.hdf5', 
                                                monitor='val_ssim', verbose=1, period=1, mode='max', save_best_only=True)
    
    # Fit the model
    model.fit(iq_dataset, dtce_dataset,
              steps_per_epoch=int(iq_count/args.bs),
              epochs=args.epochs,
              validation_data=test_dataset,
              validation_steps=int(val_count/args.bs),
              callbacks=[log_code, tensorboard, prog_bar, get_csv, saving, save_multi_model, copy_keras])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input parser
    parser.add_argument('--bs',       default= 4, type=int, help='batch size')
    parser.add_argument('--in_h',     default= 512, type=int, help='image input size height')
    parser.add_argument('--in_w',     default= 512, type=int, help='image input size width')
    parser.add_argument('--epochs',   default= 100, type=int, help='number of epochs')
    parser.add_argument('--m',        default= True, type=bool, help='manual run or hp tuning')

    # Cloud ML Params
    parser.add_argument('--job-dir', default='gs://duke-research-us/mimicknet/experiments/cyclegan_debug/{}'.format(str(time.time())), help='Job directory for Google Cloud ML')
    parser.add_argument('--image_dir', default=None, help='Local image directory')
    
    main(sys.argv)
