import tensorflow as tf
import pandas as pd
import scipy.io as sio
import numpy as np

class MimickDataset():
	def __init__(self, height=1024, width=64, log_compress=True,
				 image_dir=None, bucket_dir='gs://duke-research-us/mimicknet/data'):
		self.height = height
		self.width = width
		self.log_compress = log_compress
		self.image_dir = image_dir
		self.bucket_dir = bucket_dir

	def read_mat_generator(self):
		def read_mat_op(filename, height=self.height, width=self.width, log_compress=self.log_compress, 
						image_dir=self.image_dir, bucket_dir=self.bucket_dir):
			def _read_mat(filename):
				filepath = tf.gfile.Open('{}/{}'.format(bucket_dir, filename.decode("utf-8")), 'rb')
				if image_dir is not None:
					filepath = '{}/{}'.format(image_dir, filename.decode("utf-8"))

				matfile = sio.loadmat(filepath)
				dtce = matfile['dtce']
				dtce = (dtce - dtce.min())/(dtce.max() - dtce.min())
				
				iq = matfile['iq']
				iq = np.log10(iq) if log_compress else iq
				iq = (iq-iq.min())/(iq.max() - iq.min())
				shape = iq.shape
				
				# Pad data to batch height and width with reflections, and randomly crop
				if shape[0] < height:
					remainder = height - shape[0]
					if remainder % 2 == 0:
						dtce = np.pad(dtce, ((int(remainder/2), int(remainder/2)), (0,0)), 'reflect')
						echo = np.pad(echo, ((int(remainder/2), int(remainder/2)), (0,0)), 'reflect')
						iq = np.pad(iq, ((int(remainder/2), int(remainder/2)), (0,0)), 'reflect')
					else:
						remainder = remainder - 1
						dtce = np.pad(dtce, ((int(remainder/2) + 1, int(remainder/2)), (0,0)), 'reflect')
						echo = np.pad(echo, ((int(remainder/2) + 1, int(remainder/2)), (0,0)), 'reflect')
						iq = np.pad(iq, ((int(remainder/2) + 1, int(remainder/2)), (0,0)), 'reflect')                    
				elif shape[0] > height:
					start = np.random.randint(0, shape[0] - height)
					dtce = dtce[start:start+height, :]
					echo = echo[start:start+height, :]
					iq = iq[start:start+height, :]

				if shape[1] < width:
					remainder = width - shape[1]
					if remainder % 2 == 0:
						dtce = np.pad(dtce, ((0,0), (int(remainder/2), int(remainder/2))), 'reflect')
						echo = np.pad(echo, ((0,0), (int(remainder/2), int(remainder/2))), 'reflect')
						iq = np.pad(iq, ((0,0), (int(remainder/2), int(remainder/2))), 'reflect')
					else:
						remainder = remainder - 1
						dtce = np.pad(dtce, ((0,0), (int(remainder/2) + 1, int(remainder/2))), 'reflect')
						echo = np.pad(echo, ((0,0), (int(remainder/2) + 1, int(remainder/2))), 'reflect')
						iq = np.pad(iq, ((0,0), (int(remainder/2) + 1, int(remainder/2))), 'reflect')
				elif shape[1] > width:
					start = np.random.randint(0, shape[1] - width)
					dtce = dtce[:, start:start+width]
					iq = iq[:, start:start+width]

				return iq.astype('float32'), dtce.astype('float32')

			output = tf.py_func(_read_mat, [filename], [tf.float32, tf.float32])
			iq = tf.reshape(output[0], (height, width, 1))
			dtce = tf.reshape(output[1], (height, width, 1))
			return iq, dtce
		return read_mat_op

	def get_dataset(self, csv):
		filelist = list(pd.read_csv(csv)['filename'])
		count = len(filelist)
		dataset = tf.data.Dataset.from_tensor_slices(filelist)
		dataset = dataset.shuffle(count).repeat()
		dataset = dataset.map(self.read_mat_generator(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
		return dataset, count

	def get_paired_ultrasound_dataset(self, csv='./data/training-v1.csv', batch_size=16):
		dataset, count = self.get_dataset(csv)
		dataset = dataset.batch(batch_size).prefetch(batch_size)
		return dataset, count

	def get_unpaired_ultrasound_dataset(self, domain, csv=None, batch_size=16):
		if domain == 'iq':
			csv = './data/training_a.csv' if csv is None else csv
			dataset, count = self.get_dataset(csv)
			dataset = dataset.map(lambda iq, dtce: iq)
		
		elif domain == 'dtce':
			csv = './data/training_b.csv' if csv is None else csv
			dataset, count = get_dataset(csv)
			dataset = dataset.map(lambda iq, dtce: iq)
		else:
			raise Exception('domain must be "iq" or "dtce", given {}'.format(domain))
		
		dataset = dataset.batch(batch_size).prefetch(batch_size)
		return dataset, count

class GenerateImages(tf.keras.callbacks.Callback):
	def __init__(self, forward, log_dir, interval=1000, log_compress=True,
				 image_dir=None, bucket_dir='gs://duke-research-us/mimicknet/data', 
				 files=[('fetal', 'rfd_fetal_ch.uri_SpV5192_VpF1362_FpA6_20121101150345_1.mat'),
						('liver', 'rfd_liver_highmi.uri_SpV5192_VpF512_FpA7_20161216093626_1.mat'),
						('phantom', 'verasonics.20180206194115_channelData_part11_0.mat')]):
		super()
		self.step_count = 0
		self.interval = interval
		self.forward = forward
		self.writer = tf.summary.FileWriter(log_dir)
		self.files = []

		# Load files of interest
		for name, filename in files:
			filepath = tf.gfile.Open('{}/{}'.format(bucket_dir, filename), 'rb')
			if image_dir is not None:
				filepath = '{}/{}'.format(image_dir, filename)
				iq, dtce = mat2model(filepath, log_compress=log_compress)
				self.files.append(name, iq, dtce)

		# Create graph of metrics
		self.g1 = tf.Graph()
		with self.g1.as_default():
			self.real_placeholder = tf.placeholder(tf.float32)
			self.fake_placeholder = tf.placeholder(tf.float32)
			self.image_name = tf.placeholder(tf.string)

			mse = tf.math.reduce_mean(tf.math.square((self.real_placeholder - self.fake_placeholder)))
			mae = tf.math.reduce_mean(tf.math.abs((self.real_placeholder - self.fake_placeholder)))
			psnr = tf.image.psnr(self.real_placeholder, self.fake_placeholder, 1)
			ssim = tf.image.ssim(self.real_placeholder, self.fake_placeholder, 1)
			self.mse_summary = tf.summary.scalar(tf.strings.join(['mse', image_name], separator='_'), mse)
			self.mae_summary = tf.summary.scalar(tf.strings.join(['mae', image_name], separator='_'), mae)
			self.ssim_summary = tf.summary.scalar(tf.strings.join(['ssim', image_name], separator='_'), ssim[0])
			self.psnr_summary = tf.summary.scalar(tf.strings.join(['psnr', image_name], separator='_'), psnr[0])
			self.real_image_summary = tf.summary.image(tf.strings.join(['real', image_name], separator='_'), real_placeholder)
			self.fake_image_summary = tf.summary.image(tf.strings.join(['fake', image_name], separator='_'), fake_placeholder)
		
	def generate_images(self):
		self.step_count += 1

		# Out images on regular interval, but every 5 steps for early iterations.
		if self.step_count % self.interval == 0 or (self.step_count < 100 and self.step_count % 5 == 0):
			sess = tf.Session(graph=self.g1)
			
			# Run the forward model on the images, and write to tensorboard.
			for name, iq, dtce in self.files:
				filepath = '{}/{}'.format(self.image_dir, name)
				output = self.forward.predict(iq)
				
				mse, mae, psnr, ssim, real, fake = sess.run([self.mse_summary, 
															 self.mae_summary, 
															 self.psnr_summary, 
															 self.ssim_summary,
															 self.real_image_summary,
															 self.fake_image_summary
															 ], 
															 feed_dict = {
																self.real_placeholder: dtce,
																self.fake_placeholder: output,
																self.image_name: name
															 })

				self.writer.add_summary(mse, global_step = self.step_count)
				self.writer.add_summary(mae, global_step = self.step_count)
				self.writer.add_summary(psnr, global_step = self.step_count)
				self.writer.add_summary(ssim, global_step = self.step_count)
				self.writer.add_summary(real, global_step = self.step_count)
				self.writer.add_summary(fake, global_step = self.step_count)
				self.writer.flush()
				
	def on_batch_begin(self, batch, logs={}):
		self.generate_images()
	def on_train_end(self, logs={}):
		self.writer.close()

def ssim_multiscale(y_true, y_pred): return tf.image.ssim_multiscale(y_true, y_pred, 1)
def ssim(y_true, y_pred): return tf.image.ssim(y_true, y_pred, 1)
def psnr(y_true, y_pred): return tf.image.psnr(y_true, y_pred, 1)
def ssim_loss(y_true, y_pred): return 1-tf.image.ssim(y_true, y_pred, 1)

def combined_loss(l_ssim=0.8, l_mae=0.1, l_mse=0.1):
	def _combined_loss(y_true, y_pred):
		return l_ssim*ssim_loss(y_true, y_pred) + l_mae*tf.abs(y_true - y_pred) +  l_mse*tf.square(y_true - y_pred)
	return _combined_loss

def get_name(args, model):
	args_dict = args.__dict__
	name = model + '_'
	for key, value in args_dict.items():
		if value == True:
			name += key + ','
		elif value != False:
			name += key + '_' + str(value) + ','
	return name[:-1]

def mat2model(matfile_path, log_compress=True):
	matfile = sio.loadmat(matfile_path)
	iq = abs(matfile['iq'])
	iq = np.log10(iq) if log_compress else iq
	iq = (iq-iq.min())/(iq.max() - iq.min())
	
	dtce = matfile['dtce']
	dtce = (dtce - dtce.min())/(dtce.max() - dtce.min())
	
	shape = iq.shape

	height = iq.shape[0]
	h_pad = 0
	if height % 16 != 0:
		h_pad = 16 - height % 16

	width = iq.shape[1]
	w_pad = 0
	if width % 16 != 0:
		w_pad = 16 - width % 16
	iq = np.pad(iq, [(0, h_pad), (0, w_pad)], 'reflect')
	iq = np.expand_dims(np.expand_dims(iq, axis=0), axis=-1)
	dtce = np.pad(dtce, [(0, h_pad), (0, w_pad)], 'reflect')
	dtce = np.expand_dims(np.expand_dims(dtce, axis=0), axis=-1)
	return iq, dtce, shape
