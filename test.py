from generator import ParaclassifierGenerator
from generator import ClassifierGenerator
from keras.models import load_model
from keras import backend as K

import pandas as pd
import numpy as np
import json
import sys
import os
import gc


def build_classifier(key, dropout_rate):
    preprocess = None
    model = None
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

    model_path = os.path.join('weights', 'clf_{}_{}.h5'.format(key, dropout_rate))
    model = load_model(model_path)
    return (model, get_preprocess())


def build_paraclassifier(key):
    model = None
    model_d = {'densenet': 'dn121', 'inceptionresnet': 'irv2',
               'inception': 'iv3', 'resnet': 'r50',
               'vgg': 'vgg', 'xception': 'x'}
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

    model_path = os.path.join('weights', 'para_{}.h5'.format(key))
    model = load_model(model_path)
    return model


def classifier(config):
    results = []
    test = pd.read_csv(os.path.join('meta', 'clf_test.csv')).values.tolist()
    meta_file = open(os.path.join('meta', 'clf_meta.json'), 'r')
    min_max = json.load(meta_file)
    img_size = 256

    for key in config.keys():
        result = []
        print('Fetching Results for {} with {} dropout'.format(key, config[key]))
        df = pd.read_csv(os.path.join('logs', 'clf_{}_{}.csv'.format(key, config[key])))
        result.append(key)
        result.append(df.loc[df['val_loss'].idxmin()]['categorical_accuracy'])
        result.append(df.loc[df['val_loss'].idxmin()]['val_categorical_accuracy'])
        del df

        K.clear_session()
        model, preprocess = build_classifier(key, config[key])
        test_generator = ClassifierGenerator(preprocess, test, min_max,
                                             48, img_size, True, False)
        pred = model.evaluate_generator(test_generator)
        result.append(pred[1])
        results.append(result)

        del model
        del preprocess
        del test_generator
        _ = gc.collect()

    df = pd.DataFrame.from_records(results, columns=['Name', 'Training Accuracy',
                                                     'Validation Accuracy',
                                                     'Test Accuracy'])
    df.to_csv(os.path.join('logs', 'clf_results.csv'))
    print(df)


def paraclassifier():
    results = []
    test = pd.read_csv(os.path.join('meta', 'para_test.csv')).values.tolist()

    config = {'resnet', 'inception', 'inceptionresnet',
              'densenet', 'xception', 'vgg'}
    for key in config.keys():
        result = []
        print('Fetching Results for {}'.format(key))
        df = pd.read_csv(os.path.join('logs', 'para_{}.csv'.format(key)))
        result.append(key)
        result.append(df.loc[df['val_loss'].idxmin()]['categorical_accuracy'])
        result.append(df.loc[df['val_loss'].idxmin()]['val_categorical_accuracy'])
        del df

        K.clear_session()
        model = build_paraclassifier(key)
        test_generator = ParaclassifierGenerator(key, test, 32, True)
        pred = model.evaluate_generator(test_generator)
        result.append(pred[1])
        results.append(result)

        del model
        del test_generator
        _ = gc.collect()

    df = pd.DataFrame.from_records(results, columns=['Name', 'Training Accuracy',
                                                     'Validation Accuracy',
                                                     'Test Accuracy'])
    df.to_csv(os.path.join('logs', 'para_results.csv'))
    print(df)


if __name__ == '__main__':
    if sys.argv[1] == 'classifier':
        config = json.load(open(os.path.join('logs', 'clf_config.json'), 'r'))
        classifier(config)
    else:
        paraclassifier()
