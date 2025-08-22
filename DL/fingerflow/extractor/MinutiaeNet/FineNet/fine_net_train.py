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
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras import callbacks, preprocessing, optimizers

from . import fine_net_model

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

output_dir = '../output_FineNet/'+datetime.now().strftime('%Y%m%d-%H%M%S')

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), output_dir)
log_dir = os.path.join(os.getcwd(), output_dir + '/logs')

# Training parameters
BATCH_SIZE = 32
EPOCHS = 200
NUM_CLASSES = 2

# Model size, patch
MODEL_TYPE = 'patch224batch32'


# =============== DATA loading ========================

TRAIN_PATH = '../Dataset/train/'
TEST_PATH = '../Dataset/validate/'

input_shape = (224, 224, 3)

# Using data augmentation technique for training
datagen = preprocessing.image.ImageDataGenerator(
    # set input mean to 0 over the dataset
    featurewise_center=False,
    # set each sample mean to 0
    samplewise_center=False,
    # divide inputs by std of dataset
    featurewise_std_normalization=False,
    # divide each input by its std
    samplewise_std_normalization=False,
    # apply ZCA whitening
    zca_whitening=False,
    # randomly rotate images in the range (deg 0 to 180)
    rotation_range=180,
    # randomly shift images horizontally
    width_shift_range=0.5,
    # randomly shift images vertically
    height_shift_range=0.5,
    # randomly flip images
    horizontal_flip=True,
    # randomly flip images
    vertical_flip=True)

train_batches = datagen.flow_from_directory(
    TRAIN_PATH, target_size=(input_shape[0],
                             input_shape[1]),
    classes=['minu', 'non_minu'],
    batch_size=BATCH_SIZE)
# Feed data from directory into batches
test_gen = preprocessing.image.ImageDataGenerator()
test_batches = test_gen.flow_from_directory(
    TEST_PATH, target_size=(input_shape[0],
                            input_shape[1]),
    classes=['minu', 'non_minu'],
    batch_size=BATCH_SIZE)


# =============== end DATA loading ========================


def lr_schedule(epoch):
    """Learning Rate Schedule
    """
    l_r = 0.5e-2
    if epoch > 180:
        l_r *= 0.5e-3
    elif epoch > 150:
        l_r *= 1e-3
    elif epoch > 60:
        l_r *= 5e-2
    elif epoch > 30:
        l_r *= 5e-1
    print(('Learning rate: ', l_r))
    return l_r


# ============== Define model ==================
model = fine_net_model.get_fine_net_model(num_classes=NUM_CLASSES,
                                          pretrained_path='../Models/FineNet.h5',
                                          input_shape=input_shape)

# Save model architecture
#plot_model(model, to_file='./modelFineNet.pdf',show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
# model.summary()

# ============== End define model ==============


# ============== Other stuffs for loging and parameters ==================
MODEL_NAME = 'FineNet_%s_model.{epoch:03d}.h5' % MODEL_TYPE
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

filepath = os.path.join(save_dir, MODEL_NAME)


# Show in tensorboard
tensorboard = callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = callbacks.ModelCheckpoint(filepath=filepath,
                                       monitor='val_acc',
                                       verbose=1,
                                       save_best_only=True)

lr_scheduler = callbacks.LearningRateScheduler(lr_schedule)

lr_reducer = callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                         cooldown=0,
                                         patience=5,
                                         min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler, tensorboard]

# ============== End other stuffs  ==================

# Begin training
model.fit_generator(train_batches,
                    validation_data=test_batches,
                    epochs=EPOCHS, verbose=1,
                    callbacks=callbacks)


# Plot confusion matrix
score = model.evaluate_generator(test_batches)
print('Test accuracy:', score[1])
predictions = model.predict_generator(test_batches)
test_labels = test_batches.classes[test_batches.index_array]

cm = confusion_matrix(test_labels, np.argmax(predictions, axis=1))
cm_plot_labels = ['minu', 'non_minu']
fine_net_model.plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
