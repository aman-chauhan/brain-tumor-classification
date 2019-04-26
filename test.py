from generator import ParaclassifierGenerator
from generator import ClassifierGenerator
from keras.models import load_model
from keras import backend as K

import pandas as pd
import numpy as np
import json
import os
import gc


def build_classifier(key):
    model_path = os.path.join('weights', 'clf_{}.h5'.format(key))
    model = load_model(model_path)
    return model


def classifier(key):
    test = pd.read_csv(os.path.join('meta', 'clf_test.csv')).values.tolist()
    meta_file = open(os.path.join('meta', 'clf_meta.json'), 'r')
    min_max = json.load(meta_file)
    print('Fetching Results for {}'.format(key))
    df = pd.read_csv(os.path.join('logs', 'clf_{}.csv'.format(key)))
    acc = df.loc[df['val_loss'].idxmin()]['categorical_accuracy']
    print('Training Accuracy - {:.4f}'.format(acc))
    acc = df.loc[df['val_loss'].idxmin()]['val_categorical_accuracy']
    print('Validation Accuracy - {:.4f}'.format(acc))
    del df

    K.clear_session()
    model = build_classifier(key)
    test_generator = ClassifierGenerator(test, min_max, 4, True, False)
    pred = model.evaluate_generator(test_generator)
    print('Test Accuracy - {:.4f}'.format(acc))

    del model
    del test_generator
    _ = gc.collect()


if __name__ == '__main__':
    classifier(sys.argv[1])
