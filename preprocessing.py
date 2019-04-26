# coding: utf-8

# # Preprocessing

# ## Import Libraries

# In[ ]:


from progressbar import ProgressBar, Bar, Percentage
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from functools import reduce
import nibabel as nib
import pydicom as pdc
import pandas as pd
import numpy as np
import math
import json
import sys
import os


# ## Prepare data folder

# In[ ]:


if not os.path.exists('data'):
    os.mkdir('data')

autoencode_path = os.path.join('data', 'autoencode')
classifier_path = os.path.join('data', 'classifier')

if not os.path.exists(autoencode_path):
    os.mkdir(autoencode_path)
    os.mkdir(os.path.join(autoencode_path, 'train'))
    os.mkdir(os.path.join(autoencode_path, 'valid'))
if not os.path.exists(classifier_path):
    os.mkdir(classifier_path)
    os.mkdir(os.path.join(classifier_path, 'train'))
    os.mkdir(os.path.join(classifier_path, 'valid'))
    os.mkdir(os.path.join(classifier_path, 'test'))


# ## Seed

# In[ ]:


seed = 42


# ## Metadata Processing

# ### Normal Brain MRI

# In[ ]:


healthy_path = os.path.join('source', 'Normal')
meta = pd.read_csv(os.path.join('source', 'flipped_clinical_NormalPedBrainAge_StanfordCohort.csv'))
meta = meta[meta['Series'] == 'T2']
meta = meta[meta['Plane'] == 'Axial']
meta = meta[meta['ModelFilter'] == 'T2 ax']
meta = meta[meta['SeriesDescription'] == 'AX T2 FRFSE']
meta = meta[meta['is_Duplicate'] == 'NO']
normal_set = set(map(lambda x: os.path.join(healthy_path, '{}.npz').format(x),
                     meta['Patient_ID'].unique().tolist()))


# ### Stanford

# #### DIPG

# In[ ]:


meta = pd.read_excel(os.path.join(os.path.join('source', 'katie_annotated_metadata'),
                                  'ST_DIPG_private_all_metadata_with_roi_annotated.xlsx'))
meta = meta[meta['Series'] == 'T2']
meta = meta[meta['Plane'] == 'Axial']
meta = meta[meta['ModelFilter'] == 'T2_Axial']
meta['Filenames'] = meta['PID'] + '-' + meta['SID'].apply(lambda x: '{:02d}'.format(x)) + '-' + meta['FileName_df']
meta['Group'] = meta['PID'] + '_' + meta['FileName_df'].apply(lambda x: x.split('-')[1])
groups = meta[['Group', 'Filenames']].groupby('Group')['Filenames'].apply(sorted).values.tolist()
groups = list(filter(lambda x: all([len(y.split('-')) == 5 for y in x]) and len(x) < 51, groups))
dipg_set = set(reduce(lambda x, y: x + y, groups))


# #### EP

# In[ ]:


meta = pd.read_excel(os.path.join(os.path.join('source', 'katie_annotated_metadata'),
                                  'ST_PF-EP_private_all_metadata_with_roi_annotated.xlsx'))
meta = meta[meta['Series'] == 'T2']
meta = meta[meta['Plane'] == 'Axial']
meta = meta[meta['ModelFilter'] == 'T2_Axial']
meta['Filenames'] = meta['PID'] + '-' + meta['SID'].apply(lambda x: '{:02d}'.format(x)) + '-' + meta['FileName_df']
meta['Group'] = meta['PID'] + '_' + meta['FileName_df'].apply(lambda x: x.split('-')[1])
groups = meta[['Group', 'Filenames']].groupby('Group')['Filenames'].apply(sorted).values.tolist()
groups = list(filter(lambda x: all([len(y.split('-')) == 6 for y in x]) and len(x) < 55, groups))
ep_set = set(reduce(lambda x, y: x + y, groups))


# #### MB

# In[ ]:


meta = pd.read_excel(os.path.join(os.path.join('source', 'katie_annotated_metadata'),
                                  'ST_PF-MB_private_all_metadata_with_roi_annotated.xlsx'))
meta = meta[meta['Series'] == 'T2']
meta = meta[meta['Plane'] == 'Axial']
meta = meta[meta['ModelFilter'] == 'T2_Axial']
meta['Filenames'] = meta['PID'] + '-' + meta['SID'].apply(lambda x: '{:02d}'.format(x)) + '-' + meta['FileName_df']
meta['Group'] = meta['PID'] + '_' + meta['FileName_df'].apply(lambda x: x.split('-')[1])
groups = meta[['Group', 'Filenames']].groupby('Group')['Filenames'].apply(sorted).values.tolist()
groups = list(filter(lambda x: all([len(y.split('-')) == 6 for y in x]) and len(x) < 40, groups))
mb_set = set(reduce(lambda x, y: x + y, groups))


# #### PILO

# In[ ]:


meta = pd.read_excel(os.path.join(os.path.join('source', 'katie_annotated_metadata'),
                                  'ST_PF-PILO_private_all_metadata_with_roi_annotated.xlsx'))
meta = meta[meta['Series'] == 'T2']
meta = meta[meta['Plane'] == 'Axial']
meta = meta[meta['ModelFilter'] == 'T2_Axial']
meta['Filenames'] = meta['PID'] + '-' + meta['SID'].apply(lambda x: '{:02d}'.format(x)) + '-' + meta['FileName_df']
meta['Group'] = meta['PID'] + '_' + meta['FileName_df'].apply(lambda x: x.split('-')[1])
groups = meta[['Group', 'Filenames']].groupby('Group')['Filenames'].apply(sorted).values.tolist()
groups = list(filter(lambda x: all([len(y.split('-')) == 6 for y in x]) and len(x) < 68, groups))
pilo_set = set(reduce(lambda x, y: x + y, groups))


# ### Seattle

# #### DIPG

# In[ ]:


meta = pd.read_excel(os.path.join(os.path.join('source', 'katie_annotated_metadata'),
                                  'SE_DIPG_private_all_metadata_with_roi_annotated.xlsx'))
meta = meta[meta['Series'] == 'T2']
meta = meta[meta['Plane'] == 'Axial']
meta = meta[meta['ModelFilter'] == 'T2_Axial']
meta['Filenames'] = meta['PID'] + '-' + meta['SID'].apply(lambda x: '{:02d}'.format(x)) + '-' + meta['FileName_df']
meta['Group'] = meta['PID'] + '_' + meta['FileName_df'].apply(lambda x: x.split('-')[1])
groups = meta[['Group', 'Filenames']].groupby('Group')['Filenames'].apply(sorted).values.tolist()
groups = list(filter(lambda x: all([len(y.split('-')) == 5 for y in x]) and len(x) < 51, groups))
se_dipg_set = set(reduce(lambda x, y: x + y, groups))


# #### EP

# In[ ]:


meta = pd.read_excel(os.path.join(os.path.join('source', 'katie_annotated_metadata'),
                                  'SE_PF-EP_private_all_metadata_with_roi_annotated.xlsx'))
meta = meta[meta['Series'] == 'T2']
meta = meta[meta['Plane'] == 'Axial']
meta = meta[meta['ModelFilter'] == 'T2_Axial']
meta['Filenames'] = meta['PID'] + '-' + meta['SID'].apply(lambda x: '{:02d}'.format(x)) + '-' + meta['FileName_df']
meta['Group'] = meta['PID'] + '_' + meta['FileName_df'].apply(lambda x: x.split('-')[1])
groups = meta[['Group', 'Filenames']].groupby('Group')['Filenames'].apply(sorted).values.tolist()
groups = list(filter(lambda x: all([len(y.split('-')) == 6 for y in x]) and len(x) < 68, groups))
se_ep_set = set(reduce(lambda x, y: x + y, groups))


# #### MB

# In[ ]:


meta = pd.read_excel(os.path.join(os.path.join('source', 'katie_annotated_metadata'),
                                  'SE_PF-MB_private_all_metadata_with_roi_annotated.xlsx'))
meta = meta[meta['Series'] == 'T2']
meta = meta[meta['Plane'] == 'Axial']
meta = meta[meta['ModelFilter'] == 'T2_Axial']
meta['Filenames'] = meta['PID'] + '-' + meta['SID'].apply(lambda x: '{:02d}'.format(x)) + '-' + meta['FileName_df']
meta['Group'] = meta['PID'] + '_' + meta['FileName_df'].apply(lambda x: x.split('-')[1])
groups = meta[['Group', 'Filenames']].groupby('Group')['Filenames'].apply(sorted).values.tolist()
groups = list(filter(lambda x: all([len(y.split('-')) == 6 for y in x]) and len(x) < 68, groups))
se_mb_set = set(reduce(lambda x, y: x + y, groups))


# ## Classifier Data Preprocessing
#
# ***Using the Stanford dataset***

# In[ ]:


st_path = os.path.join(os.path.join('source', '{}'), 'Stanford')
se_path = os.path.join(os.path.join('source', '{}'), 'Seattle')

st_path = os.path.join(os.path.join(st_path, 'ST_{}_T2_Axial'), 'no_roi')
se_path = os.path.join(os.path.join(se_path, 'SE_{}_T2_Axial'), 'no_roi')
healthy_path = os.path.join('source', 'Normal')


# In[ ]:


class_files = []
class_types = []

dipg_path = st_path.format('DIPG', 'DIPG')
for _, _, files in os.walk(dipg_path):
    files = set(filter(lambda x: not x.startswith('.'), files))
    files = sorted(files.intersection(dipg_set))
    iids = {}
    for filename in files:
        key = '{}_{}'.format(filename.split('-')[0][-4:], filename.split('-')[3])
        if key not in iids:
            iids[key] = []
        iids[key].append(os.path.join(dipg_path, filename))
    files = [sorted(iids[key]) for key in sorted(iids.keys())]
    files = list(filter(lambda x: all([len(y.split('-')) == 5 for y in x]), files))
    class_files.extend(files)
    class_types.extend([0 for _ in range(len(files))])

dipg_path = se_path.format('DIPG', 'DIPG')
for _, _, files in os.walk(dipg_path):
    files = set(filter(lambda x: not x.startswith('.'), files))
    files = sorted(files.intersection(se_dipg_set))
    iids = {}
    for filename in files:
        key = '{}_{}'.format(filename.split('-')[0][-4:], filename.split('-')[3])
        if key not in iids:
            iids[key] = []
        iids[key].append(os.path.join(dipg_path, filename))
    files = [sorted(iids[key]) for key in sorted(iids.keys())]
    files = list(filter(lambda x: all([len(y.split('-')) == 5 for y in x]), files))
    class_files.extend(files)
    class_types.extend([0 for _ in range(len(files))])

ep_path = st_path.format('EP', 'PF-EP')
for _, _, files in os.walk(ep_path):
    files = set(filter(lambda x: not x.startswith('.'), files))
    files = sorted(files.intersection(ep_set))
    iids = {}
    for filename in files:
        key = '{}_{}'.format(filename.split('-')[1][-4:], filename.split('-')[4])
        if key not in iids:
            iids[key] = []
        iids[key].append(os.path.join(ep_path, filename))
    files = [sorted(iids[key]) for key in sorted(iids.keys())]
    class_files.extend(files)
    class_types.extend([1 for _ in range(len(files))])

ep_path = se_path.format('EP', 'PF-EP')
for _, _, files in os.walk(ep_path):
    files = set(filter(lambda x: not x.startswith('.'), files))
    files = sorted(files.intersection(se_ep_set))
    iids = {}
    for filename in files:
        key = '{}_{}'.format(filename.split('-')[1][-4:], filename.split('-')[4])
        if key not in iids:
            iids[key] = []
        iids[key].append(os.path.join(ep_path, filename))
    files = [sorted(iids[key]) for key in sorted(iids.keys())]
    class_files.extend(files)
    class_types.extend([1 for _ in range(len(files))])

mb_path = st_path.format('MB', 'PF-MB')
for _, _, files in os.walk(mb_path):
    files = set(filter(lambda x: not x.startswith('.'), files))
    files = sorted(files.intersection(mb_set))
    iids = {}
    for filename in files:
        key = '{}_{}'.format(filename.split('-')[1][-4:], filename.split('-')[4])
        if key not in iids:
            iids[key] = []
        iids[key].append(os.path.join(mb_path, filename))
    files = [sorted(iids[key]) for key in sorted(iids.keys())]
    class_files.extend(files)
    class_types.extend([2 for _ in range(len(files))])

mb_path = se_path.format('MB', 'PF-MB')
for _, _, files in os.walk(mb_path):
    files = set(filter(lambda x: not x.startswith('.'), files))
    files = sorted(files.intersection(se_mb_set))
    iids = {}
    for filename in files:
        key = '{}_{}'.format(filename.split('-')[1][-4:], filename.split('-')[4])
        if key not in iids:
            iids[key] = []
        iids[key].append(os.path.join(mb_path, filename))
    files = [sorted(iids[key]) for key in sorted(iids.keys())]
    class_files.extend(files)
    class_types.extend([2 for _ in range(len(files))])

pilo_path = st_path.format('PILO', 'PF-PILO')
for _, _, files in os.walk(pilo_path):
    files = set(filter(lambda x: not x.startswith('.'), files))
    files = sorted(files.intersection(pilo_set))
    iids = {}
    for filename in files:
        key = '{}_{}'.format(filename.split('-')[1][-4:], filename.split('-')[4])
        if key not in iids:
            iids[key] = []
        iids[key].append(os.path.join(pilo_path, filename))
    files = [sorted(iids[key]) for key in sorted(iids.keys())]
    class_files.extend(files)
    class_types.extend([3 for _ in range(len(files))])

for _, _, files in os.walk(healthy_path):
    files = set(map(lambda x: os.path.join(healthy_path, x),
                    filter(lambda x: not x.startswith('.'), files)))
    files = sorted(files.intersection(normal_set))
    class_files.extend(files)
    class_types.extend([4 for _ in range(len(files))])

train_path, valid_path, train_type, valid_type = train_test_split(class_files, class_types, test_size=0.2,
                                                                  random_state=seed, stratify=class_types)
valid_path, test_path, valid_type, test_type = train_test_split(valid_path, valid_type, test_size=0.5,
                                                                random_state=seed, stratify=valid_type)


# In[ ]:


class_meta = {'min': float('inf'), 'max': float('-inf')}


# In[ ]:


data_train_path = os.path.join(classifier_path, 'train')

fp = open(os.path.join('meta', 'clf_train.csv'), 'w')
fp.write('filepath, class\n')

cnt = 0
bar = ProgressBar(maxval=len(train_path), widgets=[Bar('=', '[', ']'), ' ', Percentage()]).start()

for path, method in list(zip(train_path, train_type)):
    img = None
    if method == 4:
        img = np.load(path)['T2 ax']
        dim1 = max(img.shape[:2]) - img.shape[0]
        dim2 = max(img.shape[:2]) - img.shape[1]
        if dim1 != 0 or dim2 != 0:
            pad = np.pad(img, ((math.ceil(dim1 / 2.0), math.floor(dim1 / 2.0)),
                               (math.ceil(dim2 / 2.0), math.floor(dim2 / 2.0)), (0, 0)),
                         mode='constant', constant_values=0)
            img = resize(pad, (256, 256), mode='constant', clip=True, preserve_range=True)
        img = img.transpose((2, 0, 1))
    else:
        img = []
        for filename in path:
            arr = pdc.dcmread(filename).pixel_array
            dim1 = max(arr.shape) - arr.shape[0]
            dim2 = max(arr.shape) - arr.shape[1]
            if dim1 != 0 or dim2 != 0:
                arr = np.pad(arr, ((math.ceil(dim1 / 2.0), math.floor(dim1 / 2.0)),
                                   (math.ceil(dim2 / 2.0), math.floor(dim2 / 2.0))),
                             mode='constant', constant_values=0)
            img.append(resize(arr, (256, 256), mode='constant',
                              clip=True, preserve_range=True).tolist())
        img = np.asarray(img)

    img = np.rot90(img, axes=(2, 1))
    img = (img - img.mean()) / img.std()

    min_img = img.min()
    max_img = img.max()
    if min_img < class_meta['min']:
        class_meta['min'] = min_img
    if max_img > class_meta['max']:
        class_meta['max'] = max_img

    np.savez_compressed(os.path.join(data_train_path, '{:04d}'.format(cnt)), data=img)
    fp.write('{}, {}\n'.format(os.path.join(data_train_path, '{:04d}.npz'.format(cnt)), method))
    cnt += 1
    bar.update(cnt)

bar.finish()
fp.close()


# In[ ]:


class_meta['min'] = float(class_meta['min'])
class_meta['max'] = float(class_meta['max'])
with open(os.path.join('meta', 'clf_meta.json'), 'w') as fp:
    json.dump(class_meta, fp)


# In[ ]:


data_valid_path = os.path.join(classifier_path, 'valid')

fp = open(os.path.join('meta', 'clf_valid.csv'), 'w')
fp.write('filepath, class\n')

cnt = 0
bar = ProgressBar(maxval=len(valid_path), widgets=[Bar('=', '[', ']'), ' ', Percentage()]).start()

for path, method in list(zip(valid_path, valid_type)):
    img = None
    if method == 4:
        img = np.load(path)['T2 ax']
        dim1 = max(img.shape[:2]) - img.shape[0]
        dim2 = max(img.shape[:2]) - img.shape[1]
        if dim1 != 0 or dim2 != 0:
            pad = np.pad(img, ((math.ceil(dim1 / 2.0), math.floor(dim1 / 2.0)),
                               (math.ceil(dim2 / 2.0), math.floor(dim2 / 2.0)), (0, 0)),
                         mode='constant', constant_values=0)
            img = resize(pad, (256, 256), mode='constant', clip=True, preserve_range=True)
        img = img.transpose((2, 0, 1))
    else:
        img = []
        for filename in path:
            arr = pdc.dcmread(filename).pixel_array
            dim1 = max(arr.shape) - arr.shape[0]
            dim2 = max(arr.shape) - arr.shape[1]
            if dim1 != 0 or dim2 != 0:
                arr = np.pad(arr, ((math.ceil(dim1 / 2.0), math.floor(dim1 / 2.0)),
                                   (math.ceil(dim2 / 2.0), math.floor(dim2 / 2.0))),
                             mode='constant', constant_values=0)
            img.append(resize(arr, (256, 256), mode='constant',
                              clip=True, preserve_range=True).tolist())
        img = np.asarray(img)

    img = np.rot90(img, axes=(2, 1))
    img = (img - img.mean()) / img.std()

    np.savez_compressed(os.path.join(data_valid_path, '{:04d}'.format(cnt)), data=img)
    fp.write('{}, {}\n'.format(os.path.join(data_valid_path, '{:04d}.npz'.format(cnt)), method))
    cnt += 1
    bar.update(cnt)

bar.finish()
fp.close()


# In[ ]:


data_test_path = os.path.join(classifier_path, 'test')

fp = open(os.path.join('meta', 'clf_test.csv'), 'w')
fp.write('filepath, class\n')

cnt = 0
bar = ProgressBar(maxval=len(test_path), widgets=[Bar('=', '[', ']'), ' ', Percentage()]).start()

for path, method in list(zip(test_path, test_type)):
    img = None
    if method == 4:
        img = np.load(path)['T2 ax']
        dim1 = max(img.shape[:2]) - img.shape[0]
        dim2 = max(img.shape[:2]) - img.shape[1]
        if dim1 != 0 or dim2 != 0:
            pad = np.pad(img, ((math.ceil(dim1 / 2.0), math.floor(dim1 / 2.0)),
                               (math.ceil(dim2 / 2.0), math.floor(dim2 / 2.0)), (0, 0)),
                         mode='constant', constant_values=0)
            img = resize(pad, (256, 256), mode='constant', clip=True, preserve_range=True)
        img = img.transpose((2, 0, 1))
    else:
        img = []
        for filename in path:
            arr = pdc.dcmread(filename).pixel_array
            dim1 = max(arr.shape) - arr.shape[0]
            dim2 = max(arr.shape) - arr.shape[1]
            if dim1 != 0 or dim2 != 0:
                arr = np.pad(arr, ((math.ceil(dim1 / 2.0), math.floor(dim1 / 2.0)),
                                   (math.ceil(dim2 / 2.0), math.floor(dim2 / 2.0))),
                             mode='constant', constant_values=0)
            img.append(resize(arr, (256, 256), mode='constant',
                              clip=True, preserve_range=True).tolist())
        img = np.asarray(img)

    img = np.rot90(img, axes=(2, 1))
    img = (img - img.mean()) / img.std()

    np.savez_compressed(os.path.join(data_test_path, '{:04d}'.format(cnt)), data=img)
    fp.write('{}, {}\n'.format(os.path.join(data_train_path, '{:04d}.npz'.format(cnt)), method))
    cnt += 1
    bar.update(cnt)

bar.finish()
fp.close()
