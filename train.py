from generator import ParaclassifierGenerator
from generator import AutoEncoderGenerator
from generator import ClassifierGenerator
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


def build_autoencoder(key):
    '''
    Function to build classifier model with specified key and dropout rate
    and load the model weights from logs, if already trained.

    Parameters
    ----------
    key: string
        the key to determine which transfer learning layers to use

    Returns
    -------
    tuple of Model, function and int
        returns the Model, the associated preprocessing function
        and number of epochs completed
    '''
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


def build_classifier(key, dropout_rate):
    '''
    Function to build classifier model with specified key and dropout rate
    and load the model weights from logs, if already trained.

    Parameters
    ----------
    key: string
        the key to determine which transfer learning layers to use
    dropout_rate: float
        the dropout rate for the Fully Connected Layers.

    Returns
    -------
    tuple of Model, function and int
        returns the Model, the associated preprocessing function
        and number of epochs completed
    '''
    preprocess = None
    model = None
    epoch = 0
    model_d = {'densenet': 'dn121', 'inceptionresnet': 'irv2',
               'inception': 'iv3', 'resnet': 'r50',
               'vgg': 'vgg', 'xception': 'x'}
    if key == 'densenet':
        from models.densenet import get_classifier, get_preprocess
    elif key == 'inceptionresnet':
        from models.inceptionresnet import get_classifier, get_preprocess
    elif key == 'inception':
        from models.inception import get_classifier, get_preprocess
    elif key == 'resnet':
        from models.resnet import get_classifier, get_preprocess
    elif key == 'vgg':
        from models.vgg import get_classifier, get_preprocess
    else:
        from models.xception import get_classifier, get_preprocess

    if not os.path.exists(os.path.join('weights',
                                       'clf_{}_{}.h5'.format(key, dropout_rate))):
        print('Generating new {} model.'.format(key))
        model = get_classifier(None, dropout_rate, model_d[key])
        if os.path.exists(os.path.join('weights', 'ae_{}.h5'.format(key))):
            model.load_weights(os.path.join('weights', 'ae_{}.h5'.format(key)),
                               by_name=True)
        model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
    else:
        print('Fetching {} model from logs.'.format(key))
        model_path = os.path.join('weights',
                                  'clf_{}_{}.h5'.format(key, dropout_rate))
        log_path = os.path.join('logs',
                                'clf_{}_{}.csv'.format(key, dropout_rate))
        model = load_model(model_path)
        logs = pd.read_csv(log_path)
        epoch = len(logs)
        del logs
    return (model, get_preprocess(), epoch)


def build_paraclassifier(key):
    '''
    Function to build paraclassifier model with specified key and load the model
    weights from logs.

    Parameters
    ----------
    key: string
        the key to determine which type of weights to load

    Returns
    -------
    Model
        returns the Model loaded with the weights specified by key
    '''
    model = None
    epoch = 0
    model_d = {'densenet': 'dn121', 'inceptionresnet': 'irv2',
               'inception': 'iv3', 'resnet': 'r50',
               'vgg': 'vgg', 'xception': 'x', 'ensemble': 'ensemble'}
    if key == 'densenet':
        from models.densenet import get_paraclassifier
    elif key == 'inceptionresnet':
        from models.inceptionresnet import get_paraclassifier
    elif key == 'inception':
        from models.inception import get_paraclassifier
    elif key == 'resnet':
        from models.resnet import get_paraclassifier
    elif key == 'vgg':
        from models.vgg import get_paraclassifier
    else:
        from models.xception import get_paraclassifier

    if not os.path.exists(os.path.join('weights', 'para_{}.h5'.format(key))):
        print('Generating new {} model.'.format(key))
        model = get_paraclassifier(model_d[key])
        model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
    else:
        print('Fetching {} model from logs.'.format(key))
        model_path = os.path.join('weights', 'para_{}.h5'.format(key))
        log_path = os.path.join('logs', 'para_{}.csv'.format(key))
        model = load_model(model_path)
        logs = pd.read_csv(log_path)
        epoch = len(logs)
        del logs
    return (model, epoch)


def autoencoder(key, batch_size):
    K.clear_session()
    # fetch model, preprocessing function and epochs
    model, preprocess, epoch = build_autoencoder(key)
    # training files
    train = get_filepaths(os.path.join('meta', 'ae_train.csv'))
    # validation files
    valid = get_filepaths(os.path.join('meta', 'ae_valid.csv'))
    # load the meta data for classifiers (min and max)
    meta_file = open(os.path.join('meta', 'ae_meta.json'), 'r')
    min_max = json.load(meta_file)
    img_size = 256

    # train generator
    train_generator = AutoEncoderGenerator(preprocess, train, min_max,
                                           batch_size, img_size, True, True)
    # validation generator
    valid_generator = AutoEncoderGenerator(preprocess, valid, min_max,
                                           batch_size, img_size, True, False)

    model_path = os.path.join('weights', 'ae_{}.h5'.format(key))
    log_path = os.path.join('logs', 'ae_{}.csv'.format(key))
    checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    csvlogger = CSVLogger(filename=log_path, append=True)
    # train the model
    model.fit_generator(generator=train_generator, epochs=100, verbose=1,
                        callbacks=[csvlogger, checkpoint, earlystop],
                        validation_data=valid_generator,
                        use_multiprocessing=True, workers=4,
                        initial_epoch=epoch)
    del model


def classifier(key, batch_size, dropout_rate):
    K.clear_session()
    # fetch model, preprocessing function and epochs
    model, preprocess, epoch = build_classifier(key, dropout_rate)
    # training files
    train = get_filepaths(os.path.join('meta', 'clf_train.csv'))
    # validation files
    valid = get_filepaths(os.path.join('meta', 'clf_valid.csv'))
    # load the meta data for classifiers (min and max)
    meta_file = open(os.path.join('meta', 'clf_meta.json'), 'r')
    min_max = json.load(meta_file)
    img_size = 256

    # train generator
    train_generator = ClassifierGenerator(preprocess, train, min_max,
                                          batch_size, img_size, True, True)
    # validation generator
    valid_generator = ClassifierGenerator(preprocess, valid, min_max,
                                          batch_size, img_size, True, False)

    model_path = os.path.join('weights', 'clf_{}_{}.h5'.format(key, dropout_rate))
    log_path = os.path.join('logs', 'clf_{}_{}.csv'.format(key, dropout_rate))
    checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    csvlogger = CSVLogger(filename=log_path, append=True)
    # train the model
    model.fit_generator(generator=train_generator, epochs=100, verbose=1,
                        callbacks=[csvlogger, checkpoint, earlystop],
                        validation_data=valid_generator,
                        use_multiprocessing=True, workers=4,
                        initial_epoch=epoch)
    del model


def paraclassifier(key, batch_size):
    K.clear_session()
    # fetch model and epochs
    model, epoch = build_paraclassifier(key)
    # training files
    train = get_filepaths(os.path.join('meta', 'para_train.csv'))
    # validation files
    valid = get_filepaths(os.path.join('meta', 'para_valid.csv'))
    # train generator
    train_generator = ParaclassifierGenerator(key, train, batch_size, True)
    # validation generator
    valid_generator = ParaclassifierGenerator(key, valid, batch_size, True)

    model_path = os.path.join('weights', 'para_{}.h5'.format(key))
    log_path = os.path.join('logs', 'para_{}.csv'.format(key))
    checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    csvlogger = CSVLogger(filename=log_path, append=True)
    # train the model
    model.fit_generator(generator=train_generator, epochs=100, verbose=1,
                        callbacks=[csvlogger, checkpoint, earlystop],
                        validation_data=valid_generator,
                        use_multiprocessing=True, workers=4,
                        initial_epoch=epoch)
    del model


if __name__ == '__main__':
    # create directories if they do not exist
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists('weights'):
        os.mkdir('weights')

    args = sys.argv
    if args[1] == 'autoencoder':
        autoencoder(args[2], int(args[3]))
    elif args[1] == 'classifier':
        for i in range(1, 6):
            classifier(args[2], int(args[3]), i / 10)
    elif args[1] == 'paraclassifier':
        paraclassifier(args[2], int(args[3]))
