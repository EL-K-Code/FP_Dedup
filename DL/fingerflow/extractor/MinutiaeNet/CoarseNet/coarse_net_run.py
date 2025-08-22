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
from datetime import datetime

import tensorflow as tf

from . import coarse_net_model

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# mode = 'inference'
MODE = 'deploy'

# Can use multiple folders for deploy, inference
deploy_set = ['../Dataset/CoarseNet_train/', ]
inference_set = ['../Dataset/CoarseNet_test/', ]


PRETRAIN_DIR = '../Models/CoarseNet.h5'
# output_dir = '../output_CoarseNet/'+datetime.now().strftime('%Y%m%d-%H%M%S')

FINENET_DIR = '../Models/FineNet.h5'


def main():
    if MODE == 'deploy':
        output_dir = '../output_CoarseNet/deployResults/' + datetime.now().strftime('%Y%m%d-%H%M%S')
        # logging = MinutiaeNet_utils.init_log(output_dir)
        for _, folder in enumerate(deploy_set):
            coarse_net_model.deploy_with_gt(folder, output_dir=output_dir,
                                            model_path=PRETRAIN_DIR, finenet_path=FINENET_DIR)
            # evaluate_training(model_dir=pretrain_dir, test_set=folder, logging=logging)
    elif MODE == 'inference':
        output_dir = '../output_CoarseNet/inferenceResults/' + datetime.now().strftime('%Y%m%d-%H%M%S')
        # logging = MinutiaeNet_utils.init_log(output_dir)
        for _, folder in enumerate(inference_set):
            coarse_net_model.inference(
                folder, output_dir=output_dir, model_path=PRETRAIN_DIR, finenet_path=FINENET_DIR,
                file_ext='.bmp', is_having_finenet=False)
    else:
        pass


if __name__ == '__main__':
    main()
