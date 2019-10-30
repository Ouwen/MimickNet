"""
Copyright Ouwen Huang 2019 

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import time
import argparse

def get_config():
    parser = argparse.ArgumentParser()

    # Input parser
    parser.add_argument('--bs',       default=8,    type=int, help='batch size')
    parser.add_argument('--in_h',     default=512,  type=int, help='image input size height')
    parser.add_argument('--in_w',     default=512,  type=int, help='image input size width')
    parser.add_argument('--epochs',   default=100,  type=int, help='number of epochs')
    parser.add_argument('--m',        default=True, type=bool, help='manual run or hp tuning')
    parser.add_argument('--is_test',  default=False, type=bool, help='is test')

    parser.add_argument('--train_csv', default='gs://duke-research-us/mimicknet/data/training-v2.csv', help='csv for paired training')
    parser.add_argument('--train_das_csv', default='gs://duke-research-us/mimicknet/data/training_a-v2.csv', help='csv with das images for training')
    parser.add_argument('--train_clinical_csv', default='gs://duke-research-us/mimicknet/data/training_b-v2.csv', help='csv with clinical images for training')
    parser.add_argument('--validation_csv', default='gs://duke-research-us/mimicknet/data/validation-v2.csv', help='csv for validation')
    parser.add_argument('--test_csv', default='gs://duke-research-us/mimicknet/data/testing-v2.csv', help='csv for testing')
    
    # Modeling parser
    parser.add_argument('--clipping', default=-80.0, type=float, help='DAS dB clipping')
    parser.add_argument('--kernel_height', default=3, type=int, help='height of convolution kernel')
    parser.add_argument('--cycle_consistency_loss', default=10, type=int, help='cycle consistency loss weight')
  
    # Cloud ML Params
    parser.add_argument('--job-dir', default='gs://duke-research-us/mimicknet/tmp/{}'.format(str(time.time())), help='Job directory for Google Cloud ML')
    parser.add_argument('--model_dir', default='./trained_models', help='Directory for trained models')
    parser.add_argument('--image_dir', default='gs://duke-research-us/mimicknet/data/duke-ultrasound-v1', help='Local image directory')
    
    
    parsed, unknown = parser.parse_known_args()
    
    print('Unknown args:', unknown)
    print('Parsed args:', parsed.__dict__)
    
    return parsed

config = get_config()
