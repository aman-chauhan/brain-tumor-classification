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
    '''
    Function to build classifier model with specified key and dropout rate
    and load the model weights from logs.

    Parameters
    ----------
    key: string
        the key to determine which transfer learning layers to use
    dropout_rate: float
        the dropout rate for the Fully Connected Layers.

    Returns
    -------
    tuple of Model and function
        returns the Model and the associated preprocessing function
    '''
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

    model_path = os.path.join('weights', 'para_{}.h5'.format(key))
    model = load_model(model_path)
    return model


def classifier(config):
    # list for storing results for each type of model
    results = []
    # test files
    test = pd.read_csv(os.path.join('meta', 'clf_test.csv')).values.tolist()
    # load the meta data for classifiers (min and max)
    meta_file = open(os.path.join('meta', 'clf_meta.json'), 'r')
    min_max = json.load(meta_file)
    img_size = 256

    for key in config.keys():
        result = []
        print('Fetching Results for {} with {} dropout'.format(key, config[key]))
        df = pd.read_csv(os.path.join('logs', 'clf_{}_{}.csv'.format(key, config[key])))
        # append training and validation results
        result.append(key)
        result.append(df.loc[df['val_loss'].idxmin()]['categorical_accuracy'])
        result.append(df.loc[df['val_loss'].idxmin()]['val_categorical_accuracy'])
        del df

        K.clear_session()
        # fetch model and preprocessing function
        model, preprocess = build_classifier(key, config[key])
        # test generator
        test_generator = ClassifierGenerator(preprocess, test, min_max,
                                             48, img_size, True, False)
        # fetch predictions
        pred = model.evaluate_generator(test_generator)
        # append test results
        result.append(pred[1])
        results.append(result)

        del model
        del preprocess
        del test_generator
        _ = gc.collect()

    # Create dataframe from results
    df = pd.DataFrame.from_records(results, columns=['Name', 'Training Accuracy',
                                                     'Validation Accuracy',
                                                     'Test Accuracy'])
    # Save Dataframe
    df.to_csv(os.path.join('logs', 'clf_results.csv'))
    print(df)


def paraclassifier():
    # list for storing results for each type of model
    results = []
    # test files
    test = pd.read_csv(os.path.join('meta', 'para_test.csv')).values.tolist()

    config = {'resnet', 'inception', 'inceptionresnet',
              'densenet', 'xception', 'vgg', 'ensemble'}
    for key in sorted(config):
        result = []
        print('Fetching Results for {}'.format(key))
        df = pd.read_csv(os.path.join('logs', 'para_{}.csv'.format(key)))
        # append training and validation results
        result.append(key)
        result.append(df.loc[df['val_loss'].idxmin()]['categorical_accuracy'])
        result.append(df.loc[df['val_loss'].idxmin()]['val_categorical_accuracy'])
        del df

        K.clear_session()
        # fetch model
        model = build_paraclassifier(key)
        # test generator
        test_generator = ParaclassifierGenerator(key, test, 32, True)
        # fetch predictions
        pred = model.evaluate_generator(test_generator)
        # append test results
        result.append(pred[1])
        results.append(result)

        del model
        del test_generator
        _ = gc.collect()

    # Create dataframe from results
    df = pd.DataFrame.from_records(results, columns=['Name', 'Training Accuracy',
                                                     'Validation Accuracy',
                                                     'Test Accuracy'])
    # Save Dataframe
    df.to_csv(os.path.join('logs', 'para_results.csv'))
    print(df)


if __name__ == '__main__':
    if sys.argv[1] == 'classifier':
        config = json.load(open(os.path.join('logs', 'clf_config.json'), 'r'))
        classifier(config)
    else:
        paraclassifier()
