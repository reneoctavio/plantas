# Imports
import os
import os.path
import shutil

import pandas as pd
import numpy as np

TEST_DB_PATH  = '/home/reneoctavio/Documents/Plantas/New/'
TRUE_FILES_PATH = '/home/reneoctavio/Documents/Plantas/New-Pre-Selected/'
REST_FILES_PATH = '/home/reneoctavio/Documents/Plantas/New-Manual-Selection/'

#TEST_DB_PATH  = '/media/reneoctavio/A/Plantas+/Internet/'
#TRUE_FILES_PATH = '/media/reneoctavio/A/Plantas+/Pre-Selected/'
#REST_FILES_PATH = '/media/reneoctavio/A/Plantas+/Manual-Selection/'

df = pd.read_csv(TEST_DB_PATH + 'true_files.csv')

# Create dir if not exists
if not os.path.exists(TRUE_FILES_PATH):
    os.makedirs(TRUE_FILES_PATH)

if not os.path.exists(REST_FILES_PATH):
    os.makedirs(REST_FILES_PATH)

# Read images path
for image_file in df['Files']:
    dir_name = os.path.dirname(image_file)
    label = dir_name.split(os.sep)[-1]
    file_name = os.path.basename(image_file)

    new_path = os.path.join(TRUE_FILES_PATH + label, file_name)

    # Create dir if not exists
    if not os.path.exists(TRUE_FILES_PATH + label):
        os.makedirs(TRUE_FILES_PATH + label)

    # Move
    #print(new_path)
    shutil.move(image_file, new_path)

# Copy jpeg images to be treated manually
image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(TEST_DB_PATH) for f in fn if f.endswith((".jpg", ".JPG", "*.jpeg", "*.JPEG"))]

for image_file in image_files:
    dir_name = os.path.dirname(image_file)
    label = dir_name.split(os.sep)[-1]
    file_name = os.path.basename(image_file)

    new_path = os.path.join(REST_FILES_PATH + label, file_name)

    # Create dir if not exists
    if not os.path.exists(REST_FILES_PATH + label):
        os.makedirs(REST_FILES_PATH + label)

    # Move
    #print(new_path)
    shutil.move(image_file, new_path)
