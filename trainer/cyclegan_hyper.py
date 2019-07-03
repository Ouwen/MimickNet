import tensorflow as tf
import argparse
import sys
import time
from trainer.utils import cycle_lr
from trainer import utils
from trainer import models

def main(argv):
    args = parser.parse_args()
    LOG_DIR = args.job_dir
    MODEL_DIR = '.'
    
    # Select activation function
    if args.actv == 'selu':
        Activation = tf.keras.layers.Activation(activation=tf.nn.selu)
    elif args.actv == 'lrelu':
        Activation = tf.keras.layers.LeakyReLU(args.lelu_a)
    elif args.actv == 'relu':
        Activation = tf.keras.layers.ReLU()
    else:
        raise Exception('activation must be relu, prelu, or selu')
    
    # Set filter size and shape
    filter_shape = (args.f_h, args.f_w)
    filters = [args.f1*4, args.f2*4, args.f3*4, args.f4*4, args.fbn*4]
    if args.filter_case is not None:
        filters_cases = {
            'hp_4': [16, 16, 16, 16, 16],
            'hp_8': [32, 32, 32, 32, 32],
            'hp_16': [64, 64, 64, 64, 64],
            'hp_py_2': [8, 16, 32, 64, 128],
            'hp_py_4': [16, 32, 64, 128, 256],
            'hp_py_8': [32, 64, 128, 256, 512]
        }
        filters = filters_cases[args.filter_case]
    
    # Load Data
    
    mimick_dataset = utils.MimickDataset(log_compress=args.lg_c, clipping=args.clip, image_dir=None)
    iq_dataset, iq_count = mimick_dataset.get_unpaired_ultrasound_dataset(
        domain='iq',
        csv='gs://duke-research-us/mimicknet/data/training_a.csv', 
        batch_size=args.bs, 
        shape=(args.in_h, args.in_w),
        sc=False)
    dtce_dataset, dtce_count = mimick_dataset.get_unpaired_ultrasound_dataset(
        domain='dtce',
        csv='gs://duke-research-us/mimicknet/data/training_b.csv', 
        batch_size=args.bs, 
        shape=(args.in_h, args.in_w),
        sc=False)
    test_dataset, val_count = mimick_dataset.get_paired_ultrasound_dataset(
        csv='gs://duke-research-us/mimicknet/data/testing-v1.csv', 
        batch_size=args.bs,
        shape=(args.in_h, args.in_w),
        sc=False)

    # Select and Compile Model
    if args.res:
        ModelClass = models.ResUnetModel
    else:
        ModelClass = models.UnetModel
    
    g_AB = ModelClass(shape=(None, None, 1),
                      Activation=Activation,
                      filters=filters,
                      filter_shape=filter_shape,
                      pixel_shuffler=args.ps,
                      dropout_rate=args.dr,
                      l1_regularizer=args.l1,
                      l2_regularizer=args.l2)()

    g_BA = ModelClass(shape=(None, None, 1),
                      Activation=Activation,
                      filters=filters,
                      filter_shape=filter_shape,
                      pixel_shuffler=args.ps,
                      dropout_rate=args.dr,
                      l1_regularizer=args.l1,
                      l2_regularizer=args.l2)()


    d_A = models.PatchDiscriminatorModel(shape=(args.in_h, args.in_w, 1),
                                         Activation=tf.keras.layers.ReLU(),
                                         filters=[32, 64, 128, 256, 512],
                                         filter_shape=(3,3))()

    d_B = models.PatchDiscriminatorModel(shape=(args.in_h, args.in_w, 1),
                                         Activation=tf.keras.layers.ReLU(),
                                         filters=[32, 64, 128, 256, 512],
                                         filter_shape=(3,3))()
    
    image_distance_loss = utils.combined_loss(l_ssim=args.l_ssim, l_mae=args.l_mae, l_mse=args.l_mse)
    
    model = models.CycleGAN(verbose = 1,
                            shape = (None, None, 1), 
                            g_AB=g_AB, 
                            g_BA=g_BA, 
                            d_B=d_B, 
                            d_A=d_A, 
                            patch_gan_hw=2**5)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.00002, 0.5),
                  d_loss='mse',
                  g_loss = [
                     'mse', 'mse', 
                     image_distance_loss, image_distance_loss, 
                     image_distance_loss, image_distance_loss
                  ], loss_weights = [
                     1,  1,
                     10, 10,
                     1,  1
                  ],
                  metrics=[utils.ssim, utils.psnr])
    
    # Generate Callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, write_graph=True, update_freq='epoch')
    prog_bar = tf.keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None)
    log_code = utils.LogCode(LOG_DIR, './trainer')
    
    copy_keras = utils.CopyKerasModel(MODEL_DIR, LOG_DIR)
    
    save_multi_model = utils.SaveMultiModel([('g_AB', g_AB), ('g_BA', g_BA), ('d_A', d_A), ('d_B', d_B)], MODEL_DIR)
    saving = tf.keras.callbacks.ModelCheckpoint(MODEL_DIR + '/model.{epoch:02d}-{val_ssim:.10f}.hdf5', 
                                                monitor='val_ssim', verbose=1, period=1, mode='max', save_best_only=True)
    
    image_gen = utils.GenerateImages(g_AB, LOG_DIR, log_compress=args.lg_c, clipping=args.clip, interval=int(iq_count/args.bs),
                                     files=[
                                        ('fetal', 'rfd_fetal_ch.uri_SpV5192_VpF1362_FpA6_20121101150345_1.mat'),
                                        ('fetal2', 'rfd_fetal_ch.uri_SpV6232_VpF908_FpA9_20121031150931_1.mat'),                        
                                        ('liver', 'rfd_liver_highmi.uri_SpV5192_VpF512_FpA7_20161216093626_1.mat'),
                                        ('phantom', 'verasonics.20180206194115_channelData_part11_0.mat'),
                                        ('vera_bad', 'verasonics.20170830145820_channelData_part9_1.mat'),
                                        ('sc_bad', 'sc_fetal_ch.20160909160351_sum_10.mat'),
                                        ('rfd_bad', 'rfd_liver_highmi.uri_SpV10388_VpF168_FpA8_20160901073342_2.mat')
                                     ])
    get_csv = utils.GetCsvMetrics(g_AB, LOG_DIR)
    
    # Fit the model
    model.fit(iq_dataset, dtce_dataset,
              steps_per_epoch=int(iq_count/args.bs),
              epochs=args.epochs,
              validation_data=test_dataset,
              validation_steps=int(val_count/args.bs),
              callbacks=[log_code, tensorboard, prog_bar, image_gen, get_csv, saving, save_multi_model, copy_keras])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input parser
    parser.add_argument('--bs',       default= 4, type=int, help='batch size')
    parser.add_argument('--in_h',     default= 256, type=int, help='image input size height')
    parser.add_argument('--in_w',     default= 128, type=int, help='image input size width')
    parser.add_argument('--epochs',   default= 100, type=int, help='number of epochs')
    parser.add_argument('--m',        default= True, type=bool, help='manual run or hp tuning')
    
    # Input Data Params
    parser.add_argument('--lg_c',     default= True,  type=bool, help='Log compress the raw IQ data')
    parser.add_argument('--clip',     default= True,  type=bool, help='Clip to -80 of raw beamformed data')
    
    # Activation
    parser.add_argument('--actv',     default='relu', help='activation is either relu or selu')
    parser.add_argument('--lelu_a',   default=0.1, type=float, help='lelu alpha')
    
    # ModelType
    parser.add_argument('--res',      default= True, type=bool, help='Enable residual learning')
    parser.add_argument('--ps',       default= True, type=bool, help='Enable pixel shuffler upsampling')
    
    # Model Params
    parser.add_argument('--filter_case', help='Preset filter structure')
    parser.add_argument('--f_h',      default= 3, type=int, help='filter height')
    parser.add_argument('--f_w',      default= 3, type=int, help='filter width')
    parser.add_argument('--f1',       default= 16, type=int, help='filter 1')
    parser.add_argument('--f2',       default= 16, type=int, help='filter 2')
    parser.add_argument('--f3',       default= 16, type=int, help='filter 3')
    parser.add_argument('--f4',       default= 16, type=int, help='filter 4')
    parser.add_argument('--fbn',      default= 16, type=int, help='filter bottleneck')

    # Regularization
    parser.add_argument('--dr',       default= 0, type=float, help='dropout rate')
    parser.add_argument('--l1',       default= 0.00015459984990412446, type=float, help='l1 regularization')
    parser.add_argument('--l2',       default= 0.0017282253548896074, type=float, help='l2 regularization')
    
    # Optimization Params
    parser.add_argument('--cycle_lr', default=False, type=bool,  help='cycle learning rate')
    parser.add_argument('--lr',       default=0.001, type=float, help='learning_rate')
    parser.add_argument('--l_ssim',   default=0.841, type=float, help='ssim loss')
    parser.add_argument('--l_mae',    default=0.512, type=float, help='mae loss')
    parser.add_argument('--l_mse',    default=0.110, type=float, help='mse loss')
    
    # Cloud ML Params
    parser.add_argument('--job-dir', default='gs://duke-research-us/mimicknet/experiments/cyclegan_debug/{}'.format(str(time.time())), help='Job directory for Google Cloud ML')
    
    main(sys.argv)
