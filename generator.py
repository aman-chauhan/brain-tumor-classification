# import libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence, to_categorical
import numpy as np
import os


class ClassifierGenerator(Sequence):
    def __init__(self, files, min_max, batch_size, shuffle, augment):
        self.files = files
        self.batch_size = batch_size
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
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            self.prng.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_files_temp = [self.files[k] for k in indexes]
        return self.__data_generation(list_files_temp)

    def __data_generation(self, files):
        X = []
        Y = np.empty((self.batch_size))
        for i, row in enumerate(files):
            img = np.load(row[0])['data']
            img = (img - self.min) / (self.max - self.min)
            if self.augment:
                seed = self.prng.randint(0, 1000)
                img = img.transpose((1, 2, 0)) * 255.0
                img = self.datagen.random_transform(img.astype('uint8'), seed=seed)
                img = img.transpose((2, 0, 1)) / 255.0
            X.append(np.expand_dims(img, axis=-1).tolist())
            Y[i] = row[1]
        return (pad_sequences(X, dtype='float32'),
                to_categorical(Y, num_classes=5))
