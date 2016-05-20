# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""Plantas Internet Dataset Downloader

This program downloads the Plantas Internet dataset.

Example:
    The user might set the folder to receive the images, or it will download in
    the current directory. The command should be::

        $ python internet_set_downloader.py /path/to/folder

Legal:
    Some of these images might have copyright. The use in a recognition system
    for research or non-commercial purpose constitue fair use of the data.

    This code is under the MIT license.
"""

import getopt
import gzip
import imp
import multiprocessing
import os
import os.path
import pandas as pd
import requests
import sys


def ReadCSV(destination_path):
    """Read CSV file

    Read the downloaded CSV file

    Args:
        destination_path: The path the CSV file was downloaded.

    Returns:
        A pandas table containg the images classes, URLs and file name.
    """

    # File
    compressed_file = os.path.join('data', \
        "plantas-internet.csv.gz")

    dest_file = os.path.join('data', \
        "plantas-internet.csv")

    # Decompress file
    f_in = gzip.GzipFile(compressed_file, 'rb')
    with open(dest_file, 'w') as f_out:
        f_out.write(f_in.read())
    f_in.close()

    # Read decompressed
    df = pd.read_csv(dest_file)

    return df

def Downloader(destination_path, dataframe):
    """Download images

    Download the Internet dataset images by reading URLs and putting several
    workers to download them simultaneously.

    Args:
        destination_path: Path to the directory where dataset is being saved.
        dataframe: Pandas dataframe containing image filename, URL and class.
    """
    df = dataframe
    num_of_files = len(df['File'])
    df['Dest'] = [ destination_path for i in range(0, num_of_files) ]

    num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_workers)

    input_list = [ (row['Dest'], row['File'], row['Label'], row['URL'])
                                                for _, row in df.iterrows() ]
    for i, _ in enumerate(pool.imap_unordered(_SingleWorkerDownload, input_list), 1):
        sys.stderr.write('\rdone {0:%}'.format(i / num_of_files))

def _SingleWorkerDownload(data_tuple):
    """Download a chunk of images

    Create a directory for the image class if it does not exists. Then download
    the image file and save.

    Args:
        data_tuple: Tuple containing the destination path and the image
        filename, label and URL
    """
    dest = data_tuple[0]
    filename = data_tuple[1]
    label = data_tuple[2]
    url = data_tuple[3]

    # Create species folder
    dest_path = os.path.join(dest, label)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # Download file and save
    dest_file = os.path.join(dest_path, filename)
    if not os.path.exists(dest_file):
        r = requests.get(url)
        with open(dest_file, 'wb') as f_out:
            f_out.write(r.content)

def main(destination_path):
    if not os.path.isdir(destination_path):
        raise InputError("Path does not exists")

    dataframe = ReadCSV(destination_path)
    Downloader(destination_path, dataframe)

if __name__ == "__main__":
    if (len(sys.argv) == 1):
        main(os.getcwd())
    else:
        main(sys.argv[1])
