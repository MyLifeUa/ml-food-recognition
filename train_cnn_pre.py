
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import collections

from predict import predict, predict_remote_image
from preprocessing import split_files, map_classes

print('imports success')

base_dir = './food-5/'

split_files(base_dir)
map_classes(base_dir)

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# each class will be inside here

# all images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1.0/255.)
test_datagen = ImageDataGenerator(rescale=1.0/255.)

# flow train images in batches of 20 using train_datagen
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=128,
                                                    class_mode='categorical',
                                                    shuffle=False,
                                                    target_size=(244, 244))

# flow validation images in batches of 20 using train_datagen 
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  batch_size=128,
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  target_size=(244, 244))

# import pretrained model
pretrained_model = ResNet50(
    weights = 'imagenet',include_top = False,
    input_tensor = img, input_shape = None, pooling = 'avg')

pretrained_model.summary()

# build a sequential model
model = Sequential()
model.add(InputLayer(input_shape=(244, 244, 3)))

# 1st conv block
model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
# 2nd conv block
model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())
# 3rd conv block
model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
model.add(BatchNormalization())
# ANN block
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.25))
# output layer
class_num = 5
model.add(Dense(units=class_num, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='model.2.hdf5', verbose=1, save_best_only=True)

# fit on data for 30 epochs
model.fit(train_generator, epochs=30, validation_data=test_generator, callbacks=[checkpointer])

predict_remote_image(url='https://lmld.org/wp-content/uploads/2012/07/Chocolate-Ice-Cream-3.jpg', model=model, ix_to_class=ix_to_class, debug=True)
predict_remote_image(url='https://images-gmi-pmc.edge-generalmills.com/75593ed5-420b-4782-8eae-56bdfbc2586b.jpg', model=model, ix_to_class=ix_to_class, debug=True)
