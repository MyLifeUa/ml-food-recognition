from datetime import datetime
import os
import argparse

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import ZeroPadding2D
from keras.layers import GlobalMaxPooling1D

from keras.callbacks import ModelCheckpoint, Callback
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model, get_file
from time import time

class TimingCallback(Callback):
  def __init__(self):
    self.logs=[]
  def on_epoch_begin(self,epoch, logs={}):
    self.starttime=time()
  def on_epoch_end(self,epoch, logs={}):
    self.logs.append(time()-self.starttime)


def buildClassifier( img_shape=128, num_categories=5 ):
    '''
    Builds a very simple CNN outputing num_categories.
    
    Args:
             img_shape (int): The shape of the image to feed the CNN - defaults to 128
        num_categories (int): The number of categories to feed the CNN
 
    Returns:
        keras.models.Model: a simple CNN
    
    '''
    w,h = 96,96
    classifier = Sequential()

    # Add our first convolutional layer
    classifier.add( Conv2D( filters=16,
                            kernel_size=(2,2),
                            data_format='channels_last',
                            input_shape=(w,h,3),
                            activation = 'relu',
                            name = 'firstConv2D'
                            ) )
    
    # Pooling

    # Add second convolutional layer.
    
    classifier.add( Conv2D( filters=16,
                            kernel_size=(2,2),
                            activation = 'relu',
                            name = 'secondConv2D'
                            ) 
                  )
    classifier.add( MaxPooling2D(pool_size=(2,2), name='secondMaxPool') )
    
    # Add second convolutional layer.
    
    classifier.add( Conv2D( filters=32,
                            kernel_size=(2,2),
                            activation = 'relu',
                            ) 
                  )
    

    
    classifier.add( Conv2D( filters=32,
                            kernel_size=(2,2),
                            activation = 'relu',
                            ) 
                  )

    classifier.add( Conv2D( filters=32,
                        kernel_size=(2,2),
                        activation = 'relu',
                        ) 
              )
    classifier.add( MaxPooling2D(pool_size=(2,2), name='fourthpool') )

    classifier.add( Conv2D( filters=32,
                            kernel_size=(2,2),
                            activation = 'relu',
                            ) 
                  )

    classifier.add( Conv2D( filters=32,
                        kernel_size=(2,2),
                        activation = 'relu',
                        ) 
              )
    
    """ classifier.add( ZeroPadding2D(padding=(2,2)) )
    classifier.add( Conv2D( filters=64,
                            kernel_size=(2,2),
                            activation = 'relu',
                            name = 'fifthc2'
                            ) 
                  )

    classifier.add( ZeroPadding2D(padding=(2,2)) )
    classifier.add( Conv2D( filters=128,
                            kernel_size=(2,2),
                            activation = 'relu',
                            name = 'sixthc2'
                            ) 
                  )
    classifier.add( MaxPooling2D(pool_size=(2,2), name='sixthpool') )
    
    

    classifier.add( ZeroPadding2D(padding=(2,2)) )
    classifier.add( Conv2D( filters=128,
                            kernel_size=(2,2),
                            activation = 'relu',
                            name = 'fifthc3'
                            ) 
                )

    classifier.add( ZeroPadding2D(padding=(2,2)) )
    classifier.add( Conv2D( filters=128,
                            kernel_size=(2,2),
                            activation = 'relu',
                            name = 'sixthc3'
                            ) 
                )
    classifier.add( MaxPooling2D(pool_size=(2,2), name='sixthpool3') )

 """

    # Flattening
    classifier.add( Flatten(name='flat') )

    # Add Fully connected ANN
    classifier.add( Dense( units=600, activation='relu', name='fc512') )
    classifier.add( Dense( units=250, activation='relu', name='fc256') )
    classifier.add( Dense( units=100, activation='relu', name='fc128') )
    classifier.add( Dense( units=num_categories, activation = 'softmax', name='finalfc'))

    # Compile the CNN
    #classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.summary()
    plot_model(classifier, to_file="modelv1.png", show_shapes=True, show_layer_names=False)
    return classifier


def trainModel( classifier, trainloc, testloc, img_shape, output_dir='./', batch_size=32, num_epochs=30 ):
    img_shape = (96,96) #(480,852)
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.1, 
                                       rotation_range=20,
                                       horizontal_flip = False)

    test_datagen = ImageDataGenerator(rescale = 1./255)


    training_set = train_datagen.flow_from_directory(trainloc,
                                                     target_size = img_shape,
                                                     batch_size = batch_size,
                                                     class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory(testloc,
                                                target_size = img_shape,
                                                batch_size = batch_size,
                                                class_mode = 'categorical')

    # Saves the model weights after each epoch if the validation loss decreased
    now = datetime.now()
    nowstr = now.strftime('k2tf-%Y%m%d%H%M%S')

    now = os.path.join( output_dir, nowstr)

    # Make the directory
    os.makedirs( now, exist_ok=True )

    # Create our callbacks
    savepath = os.path.join( now, 'e-{epoch:03d}-vl-{val_loss:.3f}-va-{val_acc:.3f}.h5' )
    checkpointer = ModelCheckpoint(filepath=savepath, monitor='val_acc', mode='max', verbose=0, save_best_only=True)
    fout = open( os.path.join(now, 'indices.txt'), 'wt' )
    fout.write( str(training_set.class_indices) + '\n' )

    # train the model on the new data for a few epochs
    cb = TimingCallback()
    history = classifier.fit_generator(training_set,
                             steps_per_epoch = len(training_set.filenames)//batch_size,
                             epochs = num_epochs,
                             validation_data = test_set,
                             validation_steps = len(test_set.filenames)//batch_size,
                             workers=32, 
                             max_q_size=32,
                             callbacks=[checkpointer,cb]
                             )
    
    return history,cb.logs


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    

    # Optional
    parser.add_argument('--test', dest='test', default='./validation_data', required=False, help='location of the test directory')
    parser.add_argument('--train', dest='train', default='./training_data', required=False, help='location of the test directory')
    parser.add_argument('--cats', '-c', dest='categories', default=43, type=int, required=False, help='number of categories for the model to learn')
    parser.add_argument('--output', '-o', dest='output', default='./', required=False, help='location of the output directory (default:./)')
    parser.add_argument('--batch', '-b', dest='batch', default=80, type=int, required=False, help='batch size (default:32)')
    parser.add_argument('--epochs', '-e', dest='epochs', default=20, type=int, required=False, help='number of epochs to run (default:30)')
    
    args = parser.parse_args()
    
    
    classifier = buildClassifier( 0, args.categories)
    r,logs =  trainModel( classifier, args.train, args.test, 0, args.output, batch_size=args.batch, num_epochs=args.epochs )

    plt.plot(r.history['acc'], label='acc')
    plt.plot(r.history['val_acc'], label='val_acc')
    plt.legend()
    plt.show()
    for hist,log in zip(r.history['val_acc'],logs):
      print(hist,log)
