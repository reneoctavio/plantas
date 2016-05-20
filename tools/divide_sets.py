#!/usr/bin/env python2
# Author: Rene Octavio Queiroz Dias
# GPLv3

import copy

import os
import os.path

import random
import sys

import pandas as pd
import numpy as np

def parse_csv(img_path, train=70, valid=15, test=15, core=True, expanded=True, internet=True):
    # Check if at least one is true
    if not (core or expanded or internet):
        raise ValueError('Error: At least one set should be true')

    # Check if sum is 100%
    if (train + valid + test != 100):
        raise ValueError('Error: Percentage sum must be 100%')

    # Create dataframe list
    df_list_ = []

    if core:
        df = pd.read_csv(os.path.join(img_path, 'Core' + os.sep + 'plantas-core.csv'),
                            usecols=['File', 'Label'])
        paths_ = []
        for index, row in df.iterrows():
            paths_.append(os.path.join(img_path, 'Core' + os.sep + row['Label'] + os.sep + row['File']))

        df['Path'] = paths_
        df_list_.append(df)

    if expanded:
        df = pd.read_csv(os.path.join(img_path, 'Expanded' + os.sep + 'plantas-expanded.csv'),
                            usecols=['File', 'Label'])
        paths_ = []
        for index, row in df.iterrows():
            paths_.append(os.path.join(img_path, 'Expanded' + os.sep + row['Label'] + os.sep + row['File']))

        df['Path'] = paths_
        df_list_.append(df)

    if internet:
        df = pd.read_csv(os.path.join(img_path, 'Internet' + os.sep + 'plantas-internet.csv'),
                            usecols=['File', 'Label'])
        paths_ = []
        for index, row in df.iterrows():
            paths_.append(os.path.join(img_path, 'Internet' + os.sep + row['Label'] + os.sep + row['File']))

        df['Path'] = paths_
        df_list_.append(df)

    # Create dataframe
    df = pd.concat(df_list_)
    df = df.sort_values(by='Label')
    df = df.set_index(np.arange(0, len(df.index)))

    # Get labels
    labels_ = pd.unique(df['Label'].ravel())
    print(df)

    # Count images by labels
    grouped = df.groupby('Label').count()

    # Divide
    train_sz_ = []
    valid_sz_ = []
    test_sz_  = []
    for index, row in grouped.iterrows():
        train_sz_.append(row['File'] * train / 100)
        valid_sz_.append(row['File'] * valid / 100)
        test_sz_.append(row['File'] - train_sz_[-1] - valid_sz_[-1])

    grouped['Train Size'] = train_sz_
    grouped['Valid Size'] = valid_sz_
    grouped['Test Size']  = test_sz_

    print(grouped)

    label_list = df.groupby('Label').groups
    for label, idx_list in label_list.iteritems():
        random.shuffle(idx_list)
        label_list[label] = idx_list

    label_list_path = {}
    for label, idx_list in label_list.iteritems():
        label_list_path[label] = {}
        train = grouped.loc[label]['Train Size']
        valid = grouped.loc[label]['Valid Size'] + train
        test = grouped.loc[label]['Test Size'] + valid

        print(label, train, valid, test)

        label_list_path[label]['train'] = [ os.path.join(img_path , df.iloc[idx]['Path']) for idx in idx_list[:train] ]
        label_list_path[label]['valid'] = [ os.path.join(img_path , df.iloc[idx]['Path']) for idx in idx_list[train:valid] ]
        label_list_path[label]['test']  = [ os.path.join(img_path , df.iloc[idx]['Path']) for idx in idx_list[valid:test] ]

    f_train = open(os.path.join(img_path, 'train.txt'), 'w')
    f_valid = open(os.path.join(img_path, 'valid.txt'), 'w')
    f_test = open(os.path.join(img_path, 'test.txt'), 'w')

    for label, sett in label_list_path.iteritems():
        label_num = np.where(labels_ == label)[0][0]
        for key, paths in sett.iteritems():
            if key == 'train':
                f_train.writelines(path + ' ' + str(label_num) + '\n' for path in paths)
            elif key == 'valid':
                f_valid.writelines(path + ' ' + str(label_num) + '\n' for path in paths)
            elif key == 'test':
                f_test.writelines(path + ' ' + str(label_num) + '\n' for path in paths)

    f_train.close()
    f_valid.close()
    f_test.close()

    f_label = open(os.path.join(img_path, 'label.txt'), 'w')
    f_label.writelines(label + '\n' for label in labels_)

def main(argv):
    parse_csv(img_path=argv[1])

if __name__ == "__main__":
    main(sys.argv)
