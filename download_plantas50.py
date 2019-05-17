"""Downloader for Plantas50 dataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os
import sys
import tarfile
import time
import urllib.request

# Plantas50.tar sha256 hash
P50_SHA256 = "17b35c91e3954735aa59f36555d7162058f3ff538cbd17654eb0fdaf349041e0"


def _check_hash_sha256(file, sha256hash, buffer=65536):
    """Check the hash of a file (SHA256)

    Given a file and its real hash, a hash is calculated for the file and
    compared to the given hash.

    Args:
        file: The path to a file
        sha256hash: The given hash of this file
        buffer: Size of the buffer for calculating the hash (64kb default)

    Returns:
        Whether the calculated hash is the same of the given hash.
    """
    sha256 = hashlib.sha256()
    with open(file, 'rb') as f:
        while True:
            chunk = f.read(buffer)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256hash == sha256.hexdigest()


def _progress(count, block_size, total_size):
    """Helper function that will give a download progress in stdout

    This will be the reporthook argument for the urllib.request.urlretrieve.

    Args:
        count: Count of blocks transferred so far
        block_size: A block size in bytes
        total_size: Total size of the file
    """
    global start_time
    if not count: start_time = time.time()
    elapsed_time = time.time() - start_time #s
    if elapsed_time > 60:
        min = int(elapsed_time / 60)
        sec = int(elapsed_time % 60)
        elapsed_time_str = str(min) + "m " + str(sec) + "s elapsed"
    else:
        elapsed_time_str = str(int(elapsed_time)) + "s elapsed"
    downloaded_size = float(count * block_size) / 1024. / 1024. #MB
    velocity = downloaded_size * 1024. / float(elapsed_time) # kB/s
    progress = downloaded_size * 100. / (float(total_size) / 1024. / 1024) # %
    sys.stdout.write('\r>>> Downloading %.1f%%, %.lfMB, %.lfkB/s, %s' % \
        (progress, downloaded_size, velocity, elapsed_time_str))
    sys.stdout.flush()


def _download(tarball, url, filepath):
    """Helper function that will download a Plantas50 tarball

    Args:
        tarball: The name of the tarball
        url: The URL of the tarball
        filepath: Path to the downloaded tarball
    """
    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
    statinfo = os.stat(filepath)
    print('\nDownloaded', tarball, statinfo.st_size, 'bytes.')


def download_and_merge(tarball_url_list, db_dir):
    """This function will download all tarballs and merge them

    Args:
        tarball_url_list: A file containing the tarball name, URL, SHA256 hash
        db_dir: The directory that the dataset will be saved
    """
    # Verify if Plantas50 exists and its consistency
    file = os.path.join(db_dir, 'Plantas50.tar')
    if os.path.isfile(file):
        print("Plantas50.tar already downloaded. Checking consistency...")
        if _check_hash_sha256(file, P50_SHA256):
            print("Plantas50.tar is consistent. Ready!")
            return
        else:
            print("Plantas50.tar is corrupted. Re-downloading...")
            os.remove(file)
    else:
        print("Plantas50.tar does not exists. Proceeding...")

    # Download tarballs and verify its consistency
    with open(tarball_url_list) as f:
        for line in f.readlines():
            tarball, url, sha256 = line.split()
            tarball_path = os.path.join(db_dir, tarball)
            if os.path.isfile(tarball_path):
                print(tarball + " already downloaded. Checking consistency...")
                if _check_hash_sha256(tarball_path, sha256):
                    print(tarball + " is consistent. Proceeding...")
                else:
                    print(tarball + " is corrupted. Re-downloading...")
                    os.remove(tarball_path)
                    _download(tarball, url, tarball_path)
                    print("Checking consistency...")
                    assert (_check_hash_sha256(tarball_path, sha256)), \
                    "Downloaded file is corrupted. \
                    Try again later or contact owner"
            else:
                print("Downloading " + tarball + " ...")
                _download(tarball, url, tarball_path)
                print("Checking consistency...")
                assert (_check_hash_sha256(tarball_path, sha256)), \
                "Downloaded file is corrupted. Try again later or contact owner"

    # Merge tarballs
    print("Merging tarballs into Plantas50.tar. This should take awhile...")
    tarballs_path = os.path.join(db_dir, 'Plantas50.tar.part')
    plantas50path = os.path.join(db_dir, 'Plantas50.tar')

    if os.name == 'nt':
        os.system("copy /b " + tarballs_path + "* " + plantas50path)
    else:
        os.system("cat " + tarballs_path + "* > " + plantas50path)

    if _check_hash_sha256(plantas50path, P50_SHA256):
        print("Plantas50.tar is consistent. Ready!")
    else:
        print("Plantas50.tar is corrupted. Re-run the script.")
        os.remove(plantas50path)

def main(argv):
    tarball_url_hash_file = os.path.abspath('./files_url_hashes.txt')
    db_dir = os.path.abspath('./')

    if len(argv) == 1:
        assert (os.path.isfile(tarball_url_hash_file)), "The \
        files_url_hashes.txt file was not found in the current directory. \
        Place it here or indicate its path with the command: python \
        download_plantas50.py files_url_hashes_path dataset_directory"
    elif len(argv) == 3:
        tarball_url_hash_file = os.path.abspath(argv[1])
        db_dir = os.path.abspath(argv[2])
    else:
        print("The command should be python download_plantas50.py \
        tarball_url_hash_file dataset_directory")
        return

    download_and_merge(tarball_url_hash_file, db_dir)

if __name__ == "__main__":
    main(sys.argv)
