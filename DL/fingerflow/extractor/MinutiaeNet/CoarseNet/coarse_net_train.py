"""Code for FineNet in paper "Robust Minutiae Extractor: Integrating Deep Networks and Fingerprint Domain Knowledge" at ICB 2018
  https://arxiv.org/pdf/1712.09401.pdf

  If you use whole or partial function in this code, please cite paper:

  @inproceedings{Nguyen_MinutiaeNet,
    author    = {Dinh-Luan Nguyen and Kai Cao and Anil K. Jain},
    title     = {Robust Minutiae Extractor: Integrating Deep Networks and Fingerprint Domain Knowledge},
    booktitle = {The 11th International Conference on Biometrics, 2018},
    year      = {2018},
    }
"""
import os
import argparse
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import optimizers

from . import minutiae_net_utils, coarse_net_model

os.environ['KERAS_BACKEND'] = 'tensorflow'


parser = argparse.ArgumentParser(description='Minutiae Net')
parser.add_argument('lr', type=str, default="0.005",
                    help='Setting learning rate')

parser.add_argument('GPU', type=str, default="0",
                    help='Choosing GPU')

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

BATCH_SIZE = 2
USE_MULTIPROCESSING = False
INPUT_SIZE = 400

# Can use multiple folders for training
train_set = ['../Dataset/CoarseNet_train/', ]

validate_set = ['../path/to/your/data/', ]

PRETRAIN_DIR = '../Models/CoarseNet.h5'
output_dir = '../output_CoarseNet/'+datetime.now().strftime('%Y%m%d-%H%M%S')
FINENET_DIR = '../Models/FineNet.h5'

if __name__ == '__main__':

    output_dir = '../output_CoarseNet/trainResults/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    logging = minutiae_net_utils.init_log(output_dir)
    logging.info("Learning rate = %s", args.lr)
    logging.info("Pretrain dir = %s", PRETRAIN_DIR)

    coarse_net_model.train(
        train_set=train_set, output_dir=output_dir, pretrain_dir=PRETRAIN_DIR,
        batch_size=BATCH_SIZE, test_set=validate_set,
        learning_config=optimizers.Adam(
            lr=float(args.lr),
            beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=0.9),
        logging=logging)
