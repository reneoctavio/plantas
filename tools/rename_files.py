# Imports
import os
import os.path
import shutil

import pandas as pd
import numpy as np

IMG_CORE_FILES_PATH  = '/home/reneoctavio/Documents/Plantas/Plantas+/Core/'
IMG_EXTD_FILES_PATH  = '/home/reneoctavio/Documents/Plantas/Plantas+/Expanded/'

# Core images
'''
image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(IMG_CORE_FILES_PATH) for f in fn if f.endswith((".jpg", ".JPG", "*.jpeg", "*.JPEG"))]
image_files.sort()

idx = 0
labels_ = []
filename_ = []
for image_file in image_files:
    dir_name = os.path.dirname(image_file)
    label = dir_name.split(os.sep)[-1]
    labels_.append(label)
    file_name = 'COR_' + str(idx).zfill(5) + '.JPG'
    filename_.append(file_name)
    idx += 1

    shutil.move(image_file, os.path.join(dir_name, file_name))

df = pd.DataFrame()
df['File'] = filename_
df['Label'] = labels_

df.to_csv(IMG_CORE_FILES_PATH + 'plantas-core.csv')
'''

# Extended images
image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(IMG_EXTD_FILES_PATH) for f in fn if f.endswith((".jpg", ".JPG", "*.jpeg", "*.JPEG"))]
image_files.sort()

idx = 0
labels_ = []
filename_ = []
for image_file in image_files:
    dir_name = os.path.dirname(image_file)
    label = dir_name.split(os.sep)[-1]
    labels_.append(label)
    file_name = 'EXT_' + str(idx).zfill(5) + '.JPG'
    filename_.append(file_name)
    idx += 1

    shutil.move(image_file, os.path.join(dir_name, file_name))

df = pd.DataFrame()
df['File'] = filename_
df['Label'] = labels_

df.to_csv(IMG_EXTD_FILES_PATH + 'plantas-expanded.csv')
