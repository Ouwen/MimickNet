import time

config = type('', (), {})()

config.bs = 8
config.in_h = 512
config.in_w = 512
config.epochs = 100
config.m = True
config.train_csv = 'gs://duke-research-us/mimicknet/data/training-v1.csv'
config.train_das_csv = 'gs://duke-research-us/mimicknet/data/training_a-v1.csv'
config.train_clinical_csv = 'gs://duke-research-us/mimicknet/data/training_b-v1.csv'
config.validation_csv = 'gs://duke-research-us/mimicknet/data/testing-v1.csv'
config.test_csv = 'gs://duke-research-us/mimicknet/data/testing-v1.csv'
config.job_dir = 'gs://duke-research-us/mimicknet/tmp/{}'.format(str(time.time()))
config.image_dir = 'gs://duke-research-us/mimicknet/data/duke-ultrasound-v1'
config.model_dir = './trained_models'

config.clipping = -80
config.kernel_height = 3

config.is_test = False