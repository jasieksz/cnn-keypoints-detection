#%%[markdown]
### Imports
#
import os
import numpy as np
from numpy.lib.npyio import load
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from functools import reduce

import tensorflow as tf
from tensorflow.keras.backend import flatten, sum
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, UpSampling2D, Concatenate, Flatten, GlobalAveragePooling2D
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications import ResNet50

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
def loadData(path:str, labels:list, imgSize:tuple, colorMode:str='grayscale', validationSubset:bool=False) -> tf.data.Dataset:
    batch_size = 32
    subset = "validation" if validationSubset else "training"

    return tf.keras.preprocessing.image_dataset_from_directory(
        path,
        labels=labels,
        color_mode=colorMode,
        image_size=imgSize,
        batch_size=batch_size,
        validation_split=0.2,
        subset=subset,
        seed=123
    )

def loadTestData(path:str, labels:list, imgSize:tuple, colorMode:str='grayscale') -> tf.data.Dataset:
    batch_size = 32
    return tf.keras.preprocessing.image_dataset_from_directory(
        path,
        labels=labels,
        color_mode=colorMode,
        image_size=imgSize,
        batch_size=batch_size,
    )

def loadImagesFromDir(basePath, imgSize:tuple, colorMode:str='grayscale'):
    imagePaths = [basePath + imageName for imageName in os.listdir(basePath)]
    return [tf.keras.preprocessing.image.img_to_array(img) 
            for img in [tf.keras.preprocessing.image.load_img(
                    imagePath,
                    color_mode=colorMode, 
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

def plotTrainingHistory(histories:List[tf.keras.callbacks.History], metric:str, epochs_range) -> None:
    loss = reduce(lambda a,b: a+b, [h.history[metric] for h in histories])
    val_loss = reduce(lambda a,b: a+b, [h.history['val_{}'.format(metric)] for h in histories])

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
imageH = 128
imageW = 128
numFeatures = 16

dsTrain = loadData(path='resources/{}/train'.format(objectName),
            labels=list(np.load('resources/{}/{}_keypoints_train.npy'.format(objectName, objectName)).reshape(trainDataSize, numFeatures)),
            imgSize=(imageH,imageW),
            colorMode='rgb',
            validationSubset=False)

dsValid = loadData(path='resources/{}/train'.format(objectName),
                    labels=list(np.load('resources/{}/{}_keypoints_train.npy'.format(objectName, objectName)).reshape(trainDataSize,numFeatures)),
                    imgSize=(imageH,imageW),
                    colorMode='rgb',
                    validationSubset=True)

dsTest = loadTestData(path='resources/{}/test'.format(objectName),
                    labels=list(np.load('resources/{}/{}_keypoints_test.npy'.format(objectName, objectName)).reshape(testDataSize, numFeatures)),
                    colorMode='rgb',
                    imgSize=(imageH,imageW))


#%%[markdown]
### Configure performance 
#

AUTOTUNE = tf.data.experimental.AUTOTUNE

dsTrain = dsTrain.cache().shuffle(300) \
    .prefetch(buffer_size=AUTOTUNE)

dsValid = dsValid.cache() \
    .prefetch(buffer_size=AUTOTUNE)


#%%[markdown]
### Create basic model
#
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(imageH, imageW, 3))
 
for layer in base_model.layers:
    layer.trainable = True
 
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(16)(x) 
 
model = Model(inputs=[base_model.input], outputs=[predictions])
 
model.compile(optimizer=Adam(learning_rate=0.0005),
            loss="mean_squared_error",
            metrics=['mae'])
 
model.summary()

#%%
histories = []
totalEpochs = 0

#%%[markdown]
### Train the model
#
epochs = 1000
history = model.fit(dsTrain,
                validation_data=dsValid,
                epochs=epochs)

histories.append(history)
totalEpochs += epochs


plotTrainingHistory(histories, metric='loss', epochs_range=range(totalEpochs))
plotTrainingHistory(histories, metric='mae', epochs_range=range(totalEpochs))

print("EVALUATE")
test_loss, test_acc = model.evaluate(dsTest, verbose=2)

#%%[markdown]
## Predict
#
yPred = model.predict(dsTest)
images = loadImagesFromDir('resources/{}/test/masks/'.format(objectName), (imageH, imageW), 'rgb')
labels = list(np.load('resources/{}/{}_keypoints_test.npy'.format(objectName, objectName)).reshape(testDataSize,numFeatures))

#%%[markdown]
## Plot prediction results
#
a = list(zip(images, labels, yPred))
a.sort(key=lambda t: mean_absolute_error(t[1], t[2]).numpy())
a = np.array(a)

print("BEST\n")
plotObjectsWithKeypoints(slice=0, images=a[:,0], labels=a[:,1], yPred=a[:,2])

print("\nWORST\n")
plotObjectsWithKeypoints(slice=6, images=a[:,0], labels=a[:,1], yPred=a[:,2])


#%%[markdown]
## Save model
#
if saveModel:
    model.save('resources/models/' + modelName)

#%%
model.save('resources/models/ResNet50-1010e')
