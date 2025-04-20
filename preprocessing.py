import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import pickle

class PreprocessAsKerasObject:
    '''  
    This class maintains the 80-20 split of the data from the paper we want to implement. 
    The outputs, train_prepro and val_prepro, are keras object separated into batches of self.batch_size images already. 

    In order to use the outputs, you would call model.fit(train_prepro), for example. 
    Data would be preprocessed in the same file that the model is called.
    '''

    def __init__(self):
        '''
        Creating data generators for training and validation data.
            train_datagenerator has more arguments to increase the randomness of the images for training.
        '''
        self.target_size = (256,256)
        self.batch_size = 32 # The paper used 32, but that seems insufficient given that we have 38 classes

        self.train_datagenerator = ImageDataGenerator(rescale=1./255, rotation_range=20, height_shift_range=0.1)
        self.val_datagenerator = ImageDataGenerator(rescale=1./255)

        return 
        
    def call(self, train_path, val_path):
        ''' 
        @train_path: file path to the training directory in PlantVillage (most likely ../PlantVillage/train)
        @val_path: file path to the validation directory in PlantVillage (most likely ../PlantVillage/val)
        '''

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

class PreprocessAsNumpyArrays:
    ''' 
    
    '''
    def __init__(self):
        '''
        Creating data generators for training and validation data.
            train_datagenerator has more arguments to increase the randomness of the images for training.
        '''
        self.target_size = (256,256)
        self.batch_size = 32 # The paper used 32, but that seems insufficient given that we have 38 classes

        self.train_datagenerator = ImageDataGenerator(rescale=1./255, rotation_range=20, height_shift_range=0.1)
        self.val_datagenerator = ImageDataGenerator(rescale=1./255)

        return 
        
    def call(self, train_path, val_path):
        ''' 
        @train_path: file path to the training directory in PlantVillage (most likely ../PlantVillage/train)
        @val_path: file path to the validation directory in PlantVillage (most likely ../PlantVillage/val)
        '''

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

        train_images, train_labels = train_prepro
        val_images, val_labels = val_prepro

        # Split 1/8 off of train_images and train_labels with sklearn train_test_split

        #Pickle everything

        with open("train_images.pkl", "wb") as f:
            pickle.dump(train_images, f)
        
        with open("train_labels.pkl", 'wb') as f:
            pickle.dump(train_labels, f)

        with open("val_images.pkl", 'wb') as f:
            pickle.dump(val_images, f)

