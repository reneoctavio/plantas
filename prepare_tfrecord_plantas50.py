"""Create and configure a TFRecord for Plantas50 dataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm

import io
import multiprocessing as mp
import numpy as np
import pandas as pd
import PIL
import os
import tensorflow as tf
import time
import sys
import warnings

warnings.simplefilter('ignore', UserWarning)
_RANDOM_SEED = 42

def _create_tf_example(img_data, img_format, height, width, labels):
    features = {
        'image/encoded': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[img_data])),
        'image/format': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[img_format])),
        'image/height': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[width])),
        'image/class/label': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[labels[0]])),
        'image/class/multilabel': tf.train.Feature(
            int64_list = tf.train.Int64List(value=labels))
    }
    example = tf.train.Example(features = tf.train.Features(feature = features))
    return example

def _create_example(file, label, db_dir, size, queue, aspect_ratio=None):
    # Input Size: HxW
    # PIL size: WxH
    image = PIL.Image.open(os.path.join(db_dir, file))
    new_height, new_width = size

    scale_height = float(new_height) / float(image.height)
    scale_width  = float(new_width)  / float(image.width)

    if aspect_ratio == 'min':
        scale = np.amin([scale_height, scale_width])
    elif aspect_ratio == 'max':
        scale = np.amax([scale_height, scale_width])
    elif aspect_ratio == None:
        scale = 1.
    else:
        raise ValueError('Invalid aspect_ratio option.')

    # Don't increase image size
    if scale > 1:
        scale = 1.

    new_height = int(scale * float(image.height))
    new_width  = int(scale * float(image.width))

    image = image.resize((new_width, new_height))

    # Convert image to bytes
    with io.BytesIO() as input:
        image.save(input, format='JPEG')
        image = input.getvalue()

    # Create proto example
    example = _create_tf_example(image,
                                 b'jpg',
                                 new_height,
                                 new_width,
                                 label).SerializeToString()
    # Enqueue example
    queue.put(example)

    return

def create_tfrecord(files, labels, db_dir, size, split, aspect_ratio='max'):
    # Writer
    w_file = os.path.join(db_dir, 'Plantas50/Plantas50_' + split + '.tfrecord')
    writer = tf.python_io.TFRecordWriter(w_file)
    print('Writing {}'.format(w_file))

    # Create pool and queue for parallel processing
    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool()

    # Create jobs
    jobs = []
    for file, label in zip(files, labels):
        job = pool.apply_async(_create_example, (file, label, db_dir, size,
                                                 queue, aspect_ratio))
        jobs.append(job)

    # Collect results
    for job in tqdm(jobs):
        job.get()
        example = queue.get()
        writer.write(example)
        writer.flush()

    # Guarantee nothing is left
    while not queue.empty():
        example = queue.get()
        writer.write(example)
        writer.flush()

    pool.close()
    pool.join()
    writer.close()

def split_sets(image_files, labels_file, db_dir, train_p=0.80, valid_p=0.10):
    # Read image_files and randomize rows
    df = pd.read_csv(image_files, index_col=0)
    df = df[['Path', 'Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum',
            'Kingdom']]
    df = df.sample(frac=1, random_state=_RANDOM_SEED).reset_index(drop=True)

    # Read labels file
    labels_to_ids = {}
    with open(labels_file, 'r') as f:
        for l in f.readlines():
            name, idx = l.strip().split(',')
            labels_to_ids[name] = int(idx)

    # Substitute name for id
    df[['Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum', 'Kingdom']]\
        = df[['Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum',
        'Kingdom']].applymap(lambda x: labels_to_ids[x])

    # Calculate the number of examples for each set
    num_examples = df.shape[0]
    train_ex = int(train_p * num_examples)
    valid_ex = int(valid_p * num_examples)
    test_ex  = num_examples - train_ex - valid_ex

    # Split the examples
    train = df.iloc[:train_ex]
    valid = df.iloc[train_ex:(train_ex + valid_ex)]
    test  = df.iloc[(train_ex + valid_ex):]

    files = {}
    files['train'] = train['Path'].tolist()
    files['valid'] = valid['Path'].tolist()
    files['test']  = test['Path'].tolist()

    labels = {}
    labels['train'] = train[['Species', 'Genus', 'Family', 'Order', 'Class',
        'Phylum', 'Kingdom']].values
    labels['valid'] = valid[['Species', 'Genus', 'Family', 'Order', 'Class',
        'Phylum', 'Kingdom']].values
    labels['test']  = test[['Species', 'Genus', 'Family', 'Order', 'Class',
        'Phylum', 'Kingdom']].values

    return files, labels

def save_splitted_sets_list(files, labels, dir, file):
    f = open(os.path.join(dir, 'Plantas50', file), 'w')
    for i in range(len(files)):
        f.write(os.path.join(dir, files[i]) + ' ' + str(labels[i][0]) + '\n')
    f.close()

def main(argv):
    # Read argv
    if (len(argv) >= 2) and (len(argv) <= 3):
        # Check database directory
        db_dir = os.path.abspath(argv[1])
        assert os.path.isdir(db_dir), 'Not valid directory file!'
        db_dir = os.path.split(db_dir)
        assert db_dir[1] == 'Plantas50', 'Name of directory must be Plantas50.'
        db_dir = db_dir[0]

        # Check size to resize to
        resize_to = [600, 600]
        if len(argv) == 3:
            try:
                resize_to = list(map(int, argv[2].strip().split('x')))
            except:
                raise 'Invalid entry. Must be HeightxWeight e.g. 600x600'
    else:
        raise 'Invalid number of argument, must be "path/to/Plantas50" \
              or "path/to/Plantas50 HxW"'

    # Split dataset into training, validation and test sets (80%, 10%, 10%)
    image_files = os.path.join(db_dir, 'Plantas50/Plantas50.csv')
    labels_file = os.path.join(db_dir, 'Plantas50/labels-ids.txt')
    files, labels = split_sets(image_files, labels_file, db_dir)

    # Create text lists (path and label id) / Create TFRecords files
    for split in ['train', 'valid', 'test']:
        save_splitted_sets_list(files[split], labels[split],
                                db_dir, split + '.txt')
        create_tfrecord(files[split], labels[split],
                        db_dir, resize_to, split)

    print('Your Plantas50 dataset in TFRecord is ready!')

if __name__ == "__main__":
    main(sys.argv)
