from progressbar import ProgressBar, Bar, Percentage
import pandas as pd
import numpy as np
import math
import json
import sys
import os


seed = 42


from keras.models import load_model
from keras.models import Model
from keras import backend as K
import gc


config = {'resnet': 0.4, 'inception': 0.3, 'inceptionresnet': 0.3, 'densenet': 0.2, 'xception': 0.3, 'vgg': 0.3}


paraclassifier_path = os.path.join('data', 'paraclassifier')
if not os.path.exists(paraclassifier_path):
    os.mkdir(paraclassifier_path)
    for key in config.keys():
        path = os.path.join(paraclassifier_path, key)
        os.mkdir(path)
        os.mkdir(os.path.join(path, 'train'))
        os.mkdir(os.path.join(path, 'valid'))
        os.mkdir(os.path.join(path, 'test'))


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
    model = Model(inputs=model.layers[0].input,
                  outputs=model.layers[-2].output,
                  name=key)
    return (model, get_preprocess())


df = pd.read_csv(os.path.join('meta', 'clf_train.csv'))
train_files = np.hstack((np.expand_dims(df.groupby(['filepath']).mean().index.values, axis=-1),
                         df.groupby(['filepath']).mean().values[:, 1:].astype('int'))).tolist()
df = pd.read_csv(os.path.join('meta', 'clf_valid.csv'))
valid_files = np.hstack((np.expand_dims(df.groupby(['filepath']).mean().index.values, axis=-1),
                         df.groupby(['filepath']).mean().values[:, 1:].astype('int'))).tolist()
df = pd.read_csv(os.path.join('meta', 'clf_test.csv'))
test_files = np.hstack((np.expand_dims(df.groupby(['filepath']).mean().index.values, axis=-1),
                        df.groupby(['filepath']).mean().values[:, 1:].astype('int'))).tolist()


fp = open(os.path.join('meta', 'para_train.csv'), 'w')
fp.write('filepath, class\n')
train_path = os.path.join(os.path.join(os.path.join('data', 'paraclassifier'), '{}'), 'train')
for i in range(len(train_files)):
    fp.write('{}, {}\n'.format(os.path.join(train_path, train_files[i][0].split(os.sep)[-1]), train_files[i][1]))
fp.close()

fp = open(os.path.join('meta', 'para_valid.csv'), 'w')
fp.write('filepath, class\n')
valid_path = os.path.join(os.path.join(os.path.join('data', 'paraclassifier'), '{}'), 'valid')
for i in range(len(valid_files)):
    fp.write('{}, {}\n'.format(os.path.join(valid_path, valid_files[i][0].split(os.sep)[-1]), valid_files[i][1]))
fp.close()

fp = open(os.path.join('meta', 'para_test.csv'), 'w')
fp.write('filepath, class\n')
test_path = os.path.join(os.path.join(os.path.join('data', 'paraclassifier'), '{}'), 'test')
for i in range(len(test_files)):
    fp.write('{}, {}\n'.format(os.path.join(test_path, test_files[i][0].split(os.sep)[-1]), test_files[i][1]))
fp.close()


meta_file = open(os.path.join('meta', 'clf_meta.json'), 'r')
min_max = json.load(meta_file)

for key in config.keys():
    dropout_rate = config[key]
    model, preprocess = build_classifier(key, dropout_rate)

    cnt = 0
    bar = ProgressBar(maxval=len(train_files), widgets=[Bar('=', '[', ']'), ' ', Percentage()]).start()
    for i in range(len(train_files)):
        img = np.load(train_files[i][0])['data']
        img = (img - min_max['min']) * 255.0 / (min_max['max'] - min_max['min'])
        img = np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1)
        vectors = model.predict(preprocess(img))
        np.savez_compressed(os.path.join(train_path.format(key),
                                         train_files[i][0].split(os.sep)[-1]), data=vectors)
        cnt += 1
        bar.update(cnt)
    bar.finish()

    cnt = 0
    bar = ProgressBar(maxval=len(valid_files), widgets=[Bar('=', '[', ']'), ' ', Percentage()]).start()
    for i in range(len(valid_files)):
        img = np.load(valid_files[i][0])['data']
        img = (img - min_max['min']) * 255.0 / (min_max['max'] - min_max['min'])
        img = np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1)
        vectors = model.predict(preprocess(img))
        np.savez_compressed(os.path.join(valid_path.format(key),
                                         valid_files[i][0].split(os.sep)[-1]), data=vectors)
        cnt += 1
        bar.update(cnt)
    bar.finish()

    cnt = 0
    bar = ProgressBar(maxval=len(test_files), widgets=[Bar('=', '[', ']'), ' ', Percentage()]).start()
    for i in range(len(test_files)):
        img = np.load(test_files[i][0])['data']
        img = (img - min_max['min']) * 255.0 / (min_max['max'] - min_max['min'])
        img = np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1)
        vectors = model.predict(preprocess(img))
        np.savez_compressed(os.path.join(test_path.format(key),
                                         test_files[i][0].split(os.sep)[-1]), data=vectors)
        cnt += 1
        bar.update(cnt)
    bar.finish()

    del model
    del preprocess
