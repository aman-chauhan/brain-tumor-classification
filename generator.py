# import libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence, to_categorical
import numpy as np
import os


# Classifier Generator
class ClassifierGenerator(Sequence):
    def __init__(self, pre, files, min_max, batch_size, img_size, shuffle, augment):
        '''
        Init method for AutoEncoderGenerator class

        Parameters
        ----------
        pre: function
            preprocessing function to be applied to each input
        files: list
            list of files to be read and fed in batches to the models
        min_max: dict
            dictionary of the min and max of training dataset.
            Used for Normalization.
        batch_size: int
            size of the mini-batches
        img_size: int
            size of the square image to feed the network
        shuffle: boolean
            whether to shuffle the files after each epoch
        augment: boolean
            whether to augment the individual images before
            feeding them to the models

        Returns
        -------
        None
        '''
        self.preprocess = pre
        self.files = files
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.prng = np.random.RandomState(42)
        self.datagen = ImageDataGenerator(rotation_range=45,
                                          width_shift_range=0.15,
                                          height_shift_range=0.15,
                                          shear_range=0.01,
                                          fill_mode='constant', cval=0,
                                          zoom_range=0.1)
        self.min = min_max['min']
        self.max = min_max['max']
        self.on_epoch_end()

    def on_epoch_end(self):
        '''
        Shuffles the index to files if shuffle is True

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            self.prng.shuffle(self.indexes)

    def __len__(self):
        '''
        Returns the number of mini-batches in 1 epoch

        Parameters
        ----------
        None

        Returns
        -------
        int
            the total number of mini-batches
        '''
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        '''
        Returns the (X, Y) pair of mini-batch at the i'th position

        Parameters
        ----------
        index: int
            position of the mini-batch

        Returns
        -------
        tuple
            (X, Y) pair of mini-batch at the index position
        '''
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_files_temp = [self.files[k] for k in indexes]
        return self.__data_generation(list_files_temp)

    def __data_generation(self, files):
        X = np.empty((self.batch_size, self.img_size, self.img_size, 3))
        Y = np.empty((self.batch_size))
        for i, row in enumerate(files):
            img = np.load(row[0])['data'][row[1]]
            img = (img - self.min) * 255.0 / (self.max - self.min)
            img = np.expand_dims(img, axis=-1).astype('uint8')
            if self.augment:
                seed = self.prng.randint(0, 1000)
                img = self.datagen.random_transform(img, seed=seed)
            X[i] = np.repeat(img, 3, axis=-1)
            Y[i] = row[2]
        return (self.preprocess(X), to_categorical(Y, num_classes=5))


# Paraclassifier Generator
class ParaclassifierGenerator(Sequence):
    def __init__(self, key, files, batch_size, shuffle):
        '''
        Init method for AutoEncoderGenerator class

        Parameters
        ----------
        key: string
            the model partition to read vectors from
        files: list
            list of files to be read and fed in batches to the models
        batch_size: int
            size of the mini-batches
        shuffle: boolean
            whether to shuffle the files after each epoch

        Returns
        -------
        None
        '''
        self.key = key
        self.files = files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.prng = np.random.RandomState(42)
        self.on_epoch_end()

    def on_epoch_end(self):
        '''
        Shuffles the index to files if shuffle is True

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            self.prng.shuffle(self.indexes)

    def __len__(self):
        '''
        Returns the number of mini-batches in 1 epoch

        Parameters
        ----------
        None

        Returns
        -------
        int
            the total number of mini-batches
        '''
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        '''
        Returns the (X, Y) pair of mini-batch at the i'th position

        Parameters
        ----------
        index: int
            position of the mini-batch

        Returns
        -------
        tuple
            (X, Y) pair of mini-batch at the index position
        '''
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_files_temp = [self.files[k] for k in indexes]
        return self.__data_generation(list_files_temp)

    def __data_generation(self, files):
        X = []
        Y = np.empty((self.batch_size))
        if self.key != 'ensemble':
            for i, row in enumerate(files):
                X.append(np.load(row[0].format(self.key))['data'].tolist())
                Y[i] = row[1]
        else:
            keys = ['densenet', 'inceptionresnet', 'inception',
                    'resnet', 'vgg', 'xception']
            for i, row in enumerate(files):
                T = None
                for key in keys:
                    if T is None:
                        T = np.load(row[0].format(key))['data']
                    else:
                        T += np.load(row[0].format(key))['data']
                T /= len(keys)
                X.append(T.tolist())
                Y[i] = row[1]
        return (pad_sequences(X, dtype='float32'),
                to_categorical(Y, num_classes=5))
