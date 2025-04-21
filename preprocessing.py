import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pickle

def InstallData():
    import kagglehub
    path = kagglehub.dataset_download("mohitsingh1804/plantvillage")
    print("Path to dataset files:", path)

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
    Preprocesses data and pickles train, val, and test data in a 70-20-10 split. 
    Data is saved as numpy arrays instead of a keras object.
    '''
    def __init__(self):
        '''
        Creating data generators for training and validation data.
            train_datagenerator has more arguments to increase the randomness of the images for training.
        '''
        self.target_size = (256,256)

        self.train_datagenerator = ImageDataGenerator(rescale=1./255, rotation_range=20, height_shift_range=0.1, validation_split=0.125)
        self.val_datagenerator = ImageDataGenerator(rescale=1./255) 

    def load_all_batches(self, generator):
        num_batches = int(np.ceil(generator.samples / generator.batch_size))
        images = []
        labels = []

        for _ in range(num_batches):
            im_batch, lab_batch = next(generator)
            images.append(im_batch)
            labels.append(lab_batch)

        images = tf.concat(images, axis=0)
        labels = tf.concat(labels, axis=0)

        return images, labels
        
    def call(self, train_path, val_path):
        ''' 
        @train_path: file path to the training directory in PlantVillage (most likely '../PlantVillage/train')
        @val_path: file path to the validation directory in PlantVillage (most likely '../PlantVillage/val')
        '''

        train_prepro = self.train_datagenerator.flow_from_directory(
            train_path,
            target_size=self.target_size,
            batch_size=500, #3900
            class_mode='categorical',
            shuffle=True,
            subset='training'
        )

        print("Train prepro done.")

        test_prepro = self.train_datagenerator.flow_from_directory(
            train_path,
            target_size=self.target_size,
            batch_size=500, #5500
            class_mode='categorical',
            shuffle=False, 
            subset='validation'
        )

        print('Test prepro done.')

        val_prepro = self.val_datagenerator.flow_from_directory(
            val_path,
            target_size=self.target_size,
            batch_size=500, #11000
            class_mode='categorical',
            shuffle=False
        )


        print('Validation prepro done.')

        test_images, test_labels = self.load_all_batches(test_prepro)
        print('Test data extracted.')
        val_images, val_labels = self.load_all_batches(val_prepro)
        print('Validation data extracted.')
        train_images, train_labels = self.load_all_batches(train_prepro)
        print('\nTrain data extracted.')
        

        #Pickle everything

        with open("../train_images.pkl", "wb") as f:
            pickle.dump(train_images, f)
        
        with open("../train_labels.pkl", 'wb') as f:
            pickle.dump(train_labels, f)
        print('\nTrain data pickled.')

        with open("../val_images.pkl", 'wb') as f:
            pickle.dump(val_images, f)
        
        with open("../val_labels.pkl", 'wb') as f:
            pickle.dump(val_labels, f)
        print('Validation data pickled.')

        with open("../test_images.pkl", 'wb') as f:
            pickle.dump(test_images, f)
        
        with open("../test_labels", 'wb') as f:
            pickle.dump(test_labels, f)
        print('Test data pickled.')

        return

#############################
'''
DOWNLOADING DATA

If you want to download the data from Kaggle through python, there are two options.
    Option 1: Run InstallData().
    Option 2: In terminal, type the following.
        python
        import kagglehub
        path = kagglehub.dataset_download("mohitsingh1804/plantvillage")
        print("Path to dataset files:", path)

    If you choose Option 2: path should be 'users/your_netid/.cache/kagglehub/datasets/mohitsingh1804/plantvillage/versions/1'
    This way, you can use the oscar_path variable below. If this is not the case, adjust the oscar_path accordingly.
'''

#InstallData()

#############################
'''
Make sure data is downloaded and that the paths to the data directories are correct before running the rest of the code.

If Oscar throws an error with PIL, type python preprocessor.py in terminal
'''

# For oscar
oscar_path = '../../.cache/kagglehub/datasets/mohitsingh1804/plantvillage/versions/1'
train_path = f'{oscar_path}/PlantVillage/train'
val_path = f'{oscar_path}/PlantVillage/val'

# # For local device
# train_path = '../PlantVillage/train'
# val_path = '../PlantVillage/val'

preprocessor = PreprocessAsNumpyArrays()
preprocessor.call(train_path=train_path, val_path=val_path)