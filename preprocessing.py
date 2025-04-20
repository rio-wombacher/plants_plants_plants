import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

class Preprocess:
    '''  
    This class maintains the 80-20 split of the data from the paper we want to implement.
    '''

    def __init__(self):
        '''
        Creating data generators for training and validation data.
            train_datagenerator has more arguments to increase the randomness of the images for training.
        '''
        self.target_size = (256,256)
        self.batch_size = 32

        self.train_datagenerator = ImageDataGenerator(rescale=1./255, rotation_range=20, height_shift_range=0.1)
        self.val_datagenerator = ImageDataGenerator(rescale=1./255)

        return 
        
    def call(self, train_path, val_path):

        train_prepro = self.train_datagenerator.flow_from_directory(
            train_path,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        val_prepro = self.val_datagenerator.flow_from_directory(
            val_path,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        return train_prepro, val_prepro
