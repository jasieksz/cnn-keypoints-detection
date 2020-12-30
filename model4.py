#%%[markdown]
### Imports
#
import os
import numpy as np
from numpy.lib.npyio import load
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.backend import flatten, sum
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, UpSampling2D, Concatenate
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam

#%%[markdown]
### Run config
#

saveModel = False
objectName = "drill"
modelName = "model-{}-{}".format(objectName,31)

#%%[markdown]
### Configure GPU and CUDA
#
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
config

#%%[markdown]
### Helper methods
#
def loadData(path:str, labels:list, imgSize:tuple, validationSubset:bool=False) -> tf.data.Dataset:
    batch_size = 32
    subset = "validation" if validationSubset else "training"

    return tf.keras.preprocessing.image_dataset_from_directory(
        path,
        labels=labels,
        color_mode="grayscale",
        image_size=imgSize,
        batch_size=batch_size,
        validation_split=0.2,
        subset=subset,
        seed=123
    )

def loadTestData(path:str, labels:list, imgSize:tuple) -> tf.data.Dataset:
    batch_size = 32
    return tf.keras.preprocessing.image_dataset_from_directory(
        path,
        labels=labels,
        color_mode="grayscale",
        image_size=imgSize,
        batch_size=batch_size,
    )

def loadImagesFromDir(basePath, imgSize:tuple):
    imagePaths = [basePath + imageName for imageName in os.listdir(basePath)]
    return [tf.keras.preprocessing.image.img_to_array(img) 
            for img in [tf.keras.preprocessing.image.load_img(
                    imagePath,
                    color_mode='grayscale', 
                    target_size=imgSize) for imagePath in imagePaths]]

def plotObjectsWithKeypoints(slice:int, images:list, labels:list, yPred:np.ndarray) -> None:
    plt.figure(figsize=(15,15))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(images[9*slice+i]), cmap='gray')
        plt.scatter(yPred[9*slice+i][0::2]*imageH, yPred[9*slice+i][1::2]*imageH, marker='x', s=50, c='red')
        plt.scatter(labels[9*slice+i][0::2]*imageH, labels[9*slice+i][1::2]*imageH, marker='x', s=50, c='green')
        plt.axis("off")
    plt.show()

def plotTrainingHistory(history:tf.keras.callbacks.History, metric:str) -> None:
    loss = history.history[metric]
    val_loss = history.history['val_{}'.format(metric)]

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, loss, label='Training {}'.format(metric))
    plt.plot(epochs_range, val_loss, label='Validation {}'.format(metric))
    plt.legend(loc='upper right')
    plt.title('Training and Validation {}'.format(metric))
    plt.show()

#%%[markdown]
### Load datasets
#
testDataSize = np.load('resources/{}/{}_keypoints_test.npy'.format(objectName, objectName)).shape[0]
trainDataSize = np.load('resources/{}/{}_keypoints_train.npy'.format(objectName, objectName)).shape[0]
imageH = 160
imageW = 160
numFeatures = 16

dsTrain = loadData(path='resources/{}/train'.format(objectName),
            labels=list(np.load('resources/{}/{}_keypoints_train.npy'.format(objectName, objectName)).reshape(trainDataSize, numFeatures)),
            imgSize=(imageH,imageW),
            validationSubset=False)

dsValid = loadData(path='resources/{}/train'.format(objectName),
                    labels=list(np.load('resources/{}/{}_keypoints_train.npy'.format(objectName, objectName)).reshape(trainDataSize,numFeatures)),
                    imgSize=(imageH,imageW),
                    validationSubset=True)

dsTest = loadTestData(path='resources/{}/test'.format(objectName),
                    labels=list(np.load('resources/{}/{}_keypoints_test.npy'.format(objectName, objectName)).reshape(testDataSize, numFeatures)),
                    imgSize=(imageH,imageW))


#%%[markdown]
### Configure performance 
#
# ```
# data_augmentation = tf.keras.Sequential([
#     layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
# ])
# .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE) \
# ```

AUTOTUNE = tf.data.experimental.AUTOTUNE

dsTrain = dsTrain.cache().shuffle(300) \
    .prefetch(buffer_size=AUTOTUNE)

dsValid = dsValid.cache() \
    .prefetch(buffer_size=AUTOTUNE)


#%%[markdown]
### Create basic model
#
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    intersection = sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + (1 - dice_loss(y_true, y_pred))

#%%
inputs = Input((imageH, imageW))
conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)

up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
merge6 = Concatenate(3)([drop4,up6])
conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
merge7 = Concatenate(3)([conv3,up7])
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
merge8 = Concatenate(3)([conv2,up8])
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
merge9 = Concatenate(3)([conv1, up9])
conv9 = Conv2D(48, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = Conv2D(48, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
# conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
conv10 = Dense(numFeatures)(conv9)

model = Model(inputs = inputs, outputs = conv10)

model.compile(optimizer = Adam(lr = 1e-3), loss=bce_dice_loss, metrics=[dice_loss, 'accuracy'])
model.summary()


#%%[markdown]
### Train the model
#
epochs = 10
history = model.fit(dsTrain,
                validation_data=dsValid,
                epochs=epochs)

#%%[markdown]
### Inspect training accuracy
#
plotTrainingHistory(history, metric='loss')
plotTrainingHistory(history, metric='accuracy')

#%%[markdown]
### Accuracy score model
#
test_loss, test_acc = model.evaluate(dsTest, verbose=2)

#%%[markdown]
## Predict
#
yPred = model.predict(dsTest)

#%%
images = loadImagesFromDir('resources/{}/test/masks/'.format(objectName), (imageH, imageW))
labels = list(np.load('resources/{}/{}_keypoints_test.npy'.format(objectName, objectName)).reshape(testDataSize,numFeatures))

#%%[markdown]
## Plot prediction results
#
plotObjectsWithKeypoints(slice=1, images=images, labels=labels, yPred=yPred)
plotObjectsWithKeypoints(slice=4, images=images, labels=labels, yPred=yPred)

#%%[markdown]
## Save model
#
if saveModel:
    model.save('resources/models/' + modelName)

