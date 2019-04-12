from generator import AutoEncoderGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.models import load_model
from keras import backend as K

import pandas as pd
import numpy as np
import models
import json
import sys
import os


def get_filepaths(filepath):
    return pd.read_csv(filepath).values.tolist()


def build_autoencoder(key, img_size):
    preprocess = None
    model = None
    epoch = 0
    model_d = {'densenet': 'dn121', 'inceptionresnet': 'irv2',
               'inception': 'iv3', 'resnet': 'r50',
               'vgg': 'vgg', 'xception': 'x'}
    if key == 'densenet':
        from models.densenet import get_autoencoder, get_preprocess
    elif key == 'inceptionresnet':
        from models.inceptionresnet import get_autoencoder, get_preprocess
    elif key == 'inception':
        from models.inception import get_autoencoder, get_preprocess
    elif key == 'resnet':
        from models.resnet import get_autoencoder, get_preprocess
    elif key == 'vgg':
        from models.vgg import get_autoencoder, get_preprocess
    else:
        from models.xception import get_autoencoder, get_preprocess

    if not os.path.exists(os.path.join('weights', 'ae_{}.h5'.format(key))):
        print('Generating new {} model.'.format(key))
        model = get_autoencoder(None, model_d[key])
        model.compile(optimizer='adadelta', loss='mse', metrics=['mse', 'mae'])
    else:
        print('Fetching {} model from logs.'.format(key))
        model_path = os.path.join('weights', 'ae_{}.h5'.format(key))
        log_path = os.path.join('logs', 'ae_{}.csv'.format(key))
        model = load_model(model_path)
        logs = pd.read_csv(log_path)
        epoch = len(logs)
        del logs
    return (model, get_preprocess(), epoch)


def autoencoder(key, batch_size):
    K.clear_session()
    train = get_filepaths(os.path.join('meta', 'ae_train.csv'))
    valid = get_filepaths(os.path.join('meta', 'ae_valid.csv'))
    meta_file = open(os.path.join('meta', 'ae_meta.json'), 'r')
    min_max = json.load(meta_file)
    img_size = 256
    model, preprocess, epoch = build_autoencoder(key, img_size)
    train_generator = AutoEncoderGenerator(preprocess, train, min_max,
                                           batch_size, img_size, True, True)
    valid_generator = AutoEncoderGenerator(preprocess, valid, min_max,
                                           batch_size, img_size, True, True)
    model_path = os.path.join('weights', 'ae_{}.h5'.format(key))
    log_path = os.path.join('logs', 'ae_{}.csv'.format(key))
    checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    csvlogger = CSVLogger(filename=log_path, append=True)
    model.fit_generator(generator=train_generator, epochs=100, verbose=1,
                        callbacks=[csvlogger, checkpoint, earlystop],
                        validation_data=valid_generator,
                        use_multiprocessing=True, workers=4,
                        initial_epoch=epoch)


if __name__ == '__main__':
    args = sys.argv
    if args[1] == 'autoencoder':
        autoencoder(args[2], int(args[3]))
