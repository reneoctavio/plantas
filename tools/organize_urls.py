# Imports
import os
import os.path
import shutil

import pandas as pd
import numpy as np

IMG_FILES_PATH  = '/home/reneoctavio/Documents/Plantas/Plantas+/Internet/'
CSV_FILES_PATH  = '/media/reneoctavio/221CF0C91CF098CD/Users/reneo/Downloads/Extreme Picture Finder/'

# CSV files
csv_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(CSV_FILES_PATH) for f in fn if f.endswith(".csv")]

# Read CSV files
df = pd.concat((pd.read_csv(f, sep=';', index_col=None,  names=('File', 'Path', 'URL')) for f in csv_files))
df = df.set_index(np.arange(0, len(df.index)))

# Extract Labels
labels_ = []
for index, row in df.iterrows():
    labels_.append(row['Path'].split('\\')[-2])
df['Label'] = labels_

# Try to find file in new path
not_found_idx_ = []
new_path_ = []
for index, row in df.iterrows():
    filepath = IMG_FILES_PATH + row['Label'] + os.sep + row['File']
    if (row['Label'] == 'Ixora coccinea'):
        if not os.path.isfile(filepath):
            filepath = IMG_FILES_PATH + row['Label'] + os.sep + 'compacta' + os.sep + row['File']
            if not os.path.isfile(filepath):
                not_found_idx_.append(index)
            else:
                new_path_.append(filepath)
                df.set_value(index, 'Label', 'Ixora coccinea Compacta')
        else:
            new_path_.append(filepath)
    else:
        if not os.path.isfile(filepath):
            not_found_idx_.append(index)
        else:
            new_path_.append(filepath)

df = df.drop(df.index[not_found_idx_])
df['NewPath'] = new_path_
df = df.drop_duplicates(subset='NewPath')
df = df.set_index(np.arange(0, len(df.index)))

# Create move string
new_file_name_ = []
idx = 22436
for index, row in df.iterrows():
    if row['Label'] == 'Zinnia peruviana':
        target_path_ = IMG_FILES_PATH + row['Label'] + os.sep + 'INT_' + str(idx).zfill(5) + '.JPG'
        new_file_name_.append('INT_' + str(idx).zfill(5) + '.JPG')
        print(target_path_)
        shutil.move(row['NewPath'], target_path_)
        idx += 1

new_df = pd.DataFrame()
new_df['File'] = new_file_name_
new_df['Label'] = df['Label']
new_df['URL'] = df['URL']
new_df = new_df.set_index(np.arange(22436, len(df.index) + 22436))

new_df.to_csv(IMG_FILES_PATH + 'plantas-internet-zinnia.csv')
