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

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as keras_backend

from . import coarse_net_utils


def orientation_loss(y_true, y_pred, lamb=1.):
    # clip

    y_pred = tf.clip_by_value(
        y_pred, keras_backend.epsilon(),
        1 - keras_backend.epsilon())
    # get ROI
    label_seg = keras_backend.sum(y_true, axis=-1, keepdims=True)
    label_seg = tf.cast(tf.greater(label_seg, 0), tf.float32)
    # weighted cross entropy loss
    lamb_pos, lamb_neg = 1., 1.
    logloss = lamb_pos*y_true*keras_backend.log(
        y_pred)+lamb_neg*(1-y_true)*keras_backend.log(1-y_pred)
    logloss = logloss*label_seg  # apply ROI
    logloss = -keras_backend.sum(logloss) / (keras_backend.sum(label_seg) + keras_backend.epsilon())

    # coherence loss, nearby ori should be as near as possible
    # Oritentation coherence loss

    # 3x3 ones kernel
    mean_kernal = np.reshape(
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)/8, [3, 3, 1, 1])

    sin2angle_ori, cos2angle_ori, modulus_ori = coarse_net_utils.ori2angle(y_pred)
    sin_2_angle = keras_backend.conv2d(sin2angle_ori, mean_kernal, padding='same')
    cos_2_angle = keras_backend.conv2d(cos2angle_ori, mean_kernal, padding='same')
    modulus = keras_backend.conv2d(modulus_ori, mean_kernal, padding='same')

    coherence = keras_backend.sqrt(keras_backend.square(
        sin_2_angle) + keras_backend.square(cos_2_angle)) / (modulus + keras_backend.epsilon())
    coherenceloss = keras_backend.sum(label_seg) / \
        (keras_backend.sum(coherence*label_seg) + keras_backend.epsilon()) - 1
    loss = logloss + lamb*coherenceloss
    return loss


def orientation_output_loss(y_true, y_pred):
    # clip
    y_pred = tf.clip_by_value(y_pred, keras_backend.epsilon(), 1 - keras_backend.epsilon())
    # get ROI
    label_seg = keras_backend.sum(y_true, axis=-1, keepdims=True)
    label_seg = tf.cast(tf.greater(label_seg, 0), tf.float32)
    # weighted cross entropy loss
    lamb_pos, lamb_neg = 1., 1.
    logloss = lamb_pos*y_true*keras_backend.log(
        y_pred)+lamb_neg*(1-y_true)*keras_backend.log(1-y_pred)
    logloss = logloss*label_seg  # apply ROI
    logloss = -keras_backend.sum(logloss) / (keras_backend.sum(label_seg) + keras_backend.epsilon())
    return logloss


def segmentation_loss(y_true, y_pred, lamb=1.):
    # clip
    y_pred = tf.clip_by_value(y_pred, keras_backend.epsilon(), 1 - keras_backend.epsilon())
    # weighted cross entropy loss
    total_elements = keras_backend.sum(tf.ones_like(y_true))
    label_pos = tf.cast(tf.greater(y_true, 0.0), tf.float32)
    lamb_pos = 0.5 * total_elements / keras_backend.sum(label_pos)
    lamb_neg = 1 / (2 - 1/lamb_pos)
    logloss = lamb_pos*y_true*keras_backend.log(
        y_pred)+lamb_neg*(1-y_true)*keras_backend.log(1-y_pred)
    logloss = -keras_backend.mean(keras_backend.sum(logloss, axis=-1))
    # smooth loss
    smooth_kernal = np.reshape(np.array(
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)/8, [3, 3, 1, 1])
    smoothloss = keras_backend.mean(keras_backend.abs(keras_backend.conv2d(y_pred, smooth_kernal)))
    loss = logloss + lamb*smoothloss
    return loss


def minutiae_score_loss(y_true, y_pred):
    # clip
    y_pred = tf.clip_by_value(y_pred, keras_backend.epsilon(), 1 - keras_backend.epsilon())
    # get ROI
    label_seg = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    y_true = tf.where(tf.less(y_true, 0.0),
                      tf.zeros_like(y_true), y_true)  # set -1 -> 0
    # weighted cross entropy loss
    total_elements = keras_backend.sum(label_seg) + keras_backend.epsilon()
    lamb_pos, lamb_neg = 10., .5
    logloss = lamb_pos*y_true*keras_backend.log(
        y_pred)+lamb_neg*(1-y_true)*keras_backend.log(1-y_pred)
    # apply ROI
    logloss = logloss*label_seg
    logloss = -keras_backend.sum(logloss) / total_elements
    return logloss
