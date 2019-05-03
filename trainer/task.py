import tensorflow as tf
import utils
import models

parser = argparse.ArgumentParser()

# Input parser
parser.add_argument('--bs',   default=  16, type=int, help='batch size')
parser.add_argument('--in_h', default=1024, type=int, help='image input size')
parser.add_argument('--in_w', default=  64, type=int, help='image input size')

# Input Data Params
parser.add_argument('--lg_c',  default=True,  type=bool, help='Log compress the raw IQ data')
parser.add_argument('--cmplx', default=False, type=bool, help='Complex Convolutions, Not Supported')

# Model Params
parser.add_argument('--res',  default=False, type=bool, help='Enable residual learning')
parser.add_argument('--ps',   default=False, type=bool, help='Enable pixel shuffler upsampling')
parser.add_argument('--actv', default='relu', help='activation is either relu or selu')
parser.add_argument('--f_h',  default= 3, type=int, help='filter height')
parser.add_argument('--f_w',  default= 3, type=int, help='filter width')
parser.add_argument('--f1',   default=16, type=int, help='filter 1')
parser.add_argument('--f2',   default=16, type=int, help='filter 2')
parser.add_argument('--f3',   default=16, type=int, help='filter 3')
parser.add_argument('--f4',   default=16, type=int, help='filter 4')
parser.add_argument('--fbn',  default=16, type=int, help='filter bottleneck')

# Optimization Params
parser.add_argument('--lr', default=0.002, type=float, help='learning_rate')
parser.add_argument('--l_ssim', default=0.8, type=float, help='ssim lambda')
parser.add_argument('--l_mae', default=0.1, type=float, help='ssim mae')
parser.add_argument('--l_mse', default=0.1, type=float, help='ssim mse')

LOG_DIR = 'gs://duke-research-us/mimicknet/experiments'

def main(argv):
	args = parser.parse_args()
	NAME = utils.get_name(args)
	LOG_DIR += '/' + NAME

	activation = tf.nn.selu if args.actv == 'selu' tf.nn.relu

	filter_shape = (f_h, f_w)
	filters = [args.f1, args.f2, args.f3, args.f4]
	filter_bottleneck = args.fbn

	mimick_dataset = MimickDataset(height=args.in_h, width=args.in_w, log_compress=args.lg_c)

	train_dataset, count = mimick_dataset.get_paired_ultrasound_dataset(csv='./data/training-v1.csv', batch_size=args.bs)
	test_dataset, count = mimick_dataset.get_paired_ultrasound_dataset(csv='./data/testing-v1.csv', batch_size=args.bs)

	model = models.unet(shape=(None, None, 1),
						activation=activation, 
         				filters=filters, 
         				bn_filters=filter_bottleneck, 
         				filter_shape=filter_shape, 
         				residual=args.res, 
         				pixel_shuffler=args.ps)

    model.compile(optimizer=tf.keras.optimizers.Nadam(lr=args.lr), 
    			  loss=utils.combined_loss(l_ssim=args.l_ssim, l_mae=args.l_mae, l_mse=args.l_mse),
    			  metrics=['mae', 'mse', utils.ssim, utils.ssim_multiscale, utils.psnr])
	
	tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, write_graph=True)
	saving = tf.keras.callbacks.ModelCheckpoint(LOG_DIR + '/model.{epoch:02d}-{val_loss:.10f}.hdf5', 
												monitor='val_loss', verbose=1, period=1, save_best_only=True)
	reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, min_delta=0.00001)
	image_gen = mimick_utils.GenerateImages(model, LOG_DIR, log_compress=args.lg_c, interval=int(count/args.bs/4))

	model.fit(train_dataset,
              steps_per_epoch=int(count/args.bs),
              epochs=100,
              validation_data=test_dataset,
              validation_steps=int(count/args.bs),
              verbose=1,
              callbacks=[tensorboard, saving, reduce_lr, image_gen])
