from generator import ClassifierGenerator
from model import get_classifier

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.models import load_model
from keras import backend as K

import pandas as pd
import numpy as np
import json
import sys
import os


def get_filepaths(filepath):
    return pd.read_csv(filepath).values.tolist()


def build_classifier(key):
    model = None
    epoch = 0
    if not os.path.exists(os.path.join('weights', 'clf_{}.h5'.format(key))):
        print('Generating new {} model.'.format(key))
        model = get_classifier(256, key)
        model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
    else:
        print('Fetching {} model from logs.'.format(key))
        model_path = os.path.join('weights', 'clf_{}.h5'.format(key))
        log_path = os.path.join('logs', 'clf_{}.csv'.format(key))
        model = load_model(model_path)
        logs = pd.read_csv(log_path)
        epoch = len(logs)
        del logs
    return (model, epoch)


def classifier(key, batch_size):
    K.clear_session()
    model, epoch = build_classifier(key)
    train = get_filepaths(os.path.join('meta', 'clf_train.csv'))
    valid = get_filepaths(os.path.join('meta', 'clf_valid.csv'))
    meta_file = open(os.path.join('meta', 'clf_meta.json'), 'r')
    min_max = json.load(meta_file)
    train_generator = ClassifierGenerator(train, min_max, batch_size, True, True)
    valid_generator = ClassifierGenerator(valid, min_max, batch_size, True, False)
    model_path = os.path.join('weights', 'clf_{}.h5'.format(key))
    log_path = os.path.join('logs', 'clf_{}.csv'.format(key))
    checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    csvlogger = CSVLogger(filename=log_path, append=True)
    model.fit_generator(generator=train_generator, epochs=100, verbose=1,
                        callbacks=[csvlogger, checkpoint, earlystop],
                        validation_data=valid_generator,
                        use_multiprocessing=False,
                        initial_epoch=epoch)
    del model


if __name__ == '__main__':
    classifier(sys.argv[1], int(sys.argv[2]))
