import tensorflow as tf
import argparse
from trainer import cycle_lr
import sys
from trainer import utils
from trainer import models

def main(argv):
    args = parser.parse_args()
    LOG_DIR = args.job_dir + '/' + utils.get_name(args, 'unet') if args.m else args.job_dir
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
    
    # Load Data
    mimick_dataset = utils.MimickDataset(height=args.in_h, width=args.in_w, log_compress=args.lg_c)
    train_dataset, count = mimick_dataset.get_paired_ultrasound_dataset(csv='gs://duke-research-us/mimicknet/data/training-v1.csv', batch_size=args.bs)
    test_dataset, count = mimick_dataset.get_paired_ultrasound_dataset(csv='gs://duke-research-us/mimicknet/data/testing-v1.csv', batch_size=args.bs)
    
    # Select and Compile Model
    if args.res:
        ModelClass = models.ResUnetModel
    else:
        ModelClass = models.UnetModel
    
    model = ModelClass(shape=(None, None, 1),
                       Activation=Activation,
                       filters=filters,
                       filter_shape=filter_shape,
                       pixel_shuffler=args.ps,
                       dropout_rate=args.dr,
                       l1_regularizer=args.l1,
                       l2_regularizer=args.l2)()
    
    model.compile(optimizer=tf.keras.optimizers.Nadam(lr=args.lr),
                  loss=utils.combined_loss(l_ssim=args.l_ssim, l_mae=args.l_mae, l_mse=args.l_mse),
                  metrics=['mae', 'mse', utils.ssim, utils.psnr])
    
    # Generate Callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, write_graph=True, update_freq='epoch')
    saving = tf.keras.callbacks.ModelCheckpoint(MODEL_DIR + '/model.{epoch:02d}-{val_ssim:.10f}.hdf5', 
                                                monitor='val_ssim', verbose=1, period=1, save_best_only=True)
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_ssim', patience=8, verbose=1, mode='auto', restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=args.lr*.01)
    if args.cycle_lr:
        reduce_lr = cycle_lr.CyclicLR(base_lr=args.lr*.01, max_lr=args.lr*10, step_size=1000, mode='triangular2')

    image_gen = utils.GenerateImages(model, LOG_DIR, log_compress=args.lg_c, interval=int(count/args.bs),
                                     files=[
                                        ('fetal', 'rfd_fetal_ch.uri_SpV5192_VpF1362_FpA6_20121101150345_1.mat'),
                                        ('fetal2', 'rfd_fetal_ch.uri_SpV6232_VpF908_FpA9_20121031150931_1.mat'),                        
                                        ('liver', 'rfd_liver_highmi.uri_SpV5192_VpF512_FpA7_20161216093626_1.mat'),
                                        ('liver2', 'rfd_liver_highmi.uri_SpV5192_VpF512_FpA7_20161216093626_1.mat'),
                                        ('phantom', 'verasonics.20180206194115_channelData_part11_0.mat'),
                                        ('vera_bad', 'verasonics.20170830145820_channelData_part9_1.mat'),
                                        ('sc_bad', 'sc_fetal_ch.20160909160351_sum_10.mat'),
                                        ('rfd_bad', 'rfd_liver_highmi.uri_SpV10388_VpF168_FpA8_20160901073342_2.mat')
                                     ])
    
    copy_keras = utils.CopyKerasModel(MODEL_DIR, LOG_DIR)
    terminate = tf.keras.callbacks.TerminateOnNaN()
    
    # Fit the model
    model.fit(train_dataset,
              steps_per_epoch=int(count/args.bs),
              epochs=args.epochs,
              validation_data=test_dataset,
              validation_steps=int(count/args.bs),
              verbose=1,
              callbacks=[terminate, tensorboard, saving, reduce_lr, copy_keras, image_gen, early_stop])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input parser
    parser.add_argument('--bs',       default= 32, type=int, help='batch size')
    parser.add_argument('--in_h',     default= 512, type=int, help='image input size height')
    parser.add_argument('--in_w',     default= 64, type=int, help='image input size width')
    parser.add_argument('--epochs',   default= 100, type=int, help='number of epochs')
    parser.add_argument('--m',        default= True, type=bool, help='manual run or hp tuning')
    
    # Input Data Params
    parser.add_argument('--lg_c',     default= False,  type=bool, help='Log compress the raw IQ data')
    
    # Activation
    parser.add_argument('--actv',     default='relu', help='activation is either relu or selu')
    parser.add_argument('--lelu_a',   default=0.1, type=float, help='activation is either relu or selu')
    
    # ModelType
    parser.add_argument('--res',      default= False, type=bool, help='Enable residual learning')
    parser.add_argument('--ps',       default= False, type=bool, help='Enable pixel shuffler upsampling')

    # Model Params
    parser.add_argument('--f_h',      default= 3, type=int, help='filter height')
    parser.add_argument('--f_w',      default= 3, type=int, help='filter width')
    parser.add_argument('--f1',       default= 4, type=int, help='filter 1')
    parser.add_argument('--f2',       default= 4, type=int, help='filter 1')
    parser.add_argument('--f3',       default= 4, type=int, help='filter 1')
    parser.add_argument('--f4',       default= 4, type=int, help='filter 1')
    parser.add_argument('--fbn',      default= 4, type=int, help='filter 1')

    # Regularization
    parser.add_argument('--dr',       default= 0, type=float, help='dropout rate')
    parser.add_argument('--l1',       default= 0, type=float, help='dropout rate')
    parser.add_argument('--l2',       default= 0, type=float, help='dropout rate')
    
    # Optimization Params
    parser.add_argument('--cycle_lr', default=False, type=bool,  help='cycle learning rate')
    parser.add_argument('--lr',       default=0.002, type=float, help='learning_rate')
    parser.add_argument('--l_ssim',   default=True,  type=bool, help='ssim lambda')
    parser.add_argument('--l_mae',    default=False, type=bool, help='ssim mae')
    parser.add_argument('--l_mse',    default=False, type=bool, help='ssim mse')

    # Cloud ML Params
    parser.add_argument('--job-dir', default='gs://duke-research-us/mimicknet/experiments/manual_unet', help='Job directory for Google Cloud ML')
    
    main(sys.argv)
