"""Downloader for Plantas50 dataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os
import sys
import tarfile
import time

try:
    import urllib.request as urlrequest
except ImportError:
    import urllib as urlrequest

# Plantas50.tar sha256 hash
P50_SHA256 = "af66a7625fcbdc1082ea197c2d4fb6514d362c4e04772bfc06bc7a0f5435a550"

# download_file_list.tar.bz2 sha256 hash and download URL
DLF_SHA256 = "aacf87348804aa3ff6b9afab2ed8f04cbcaac8433e2d7f09c889440b0fc7505e"
DLF_URL = "https://github.com/reneoctavio/plantas/raw/master/download_file_list.tar.bz2"

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
    if progress > 100.:
        progress = 100.
    sys.stdout.write('\r>>> Downloading %.1f%%, %.lfMB, %.lfkB/s, %s' % \
        (progress, downloaded_size, velocity, elapsed_time_str))
    sys.stdout.flush()


def _download(url, filepath):
    """Helper function that will download a Plantas50 tarball

    Args:
        url: The URL of the tarball
        filepath: Path to the downloaded tarball
    """
    filepath, _ = urlrequest.urlretrieve(url, filepath, _progress)
    statinfo = os.stat(filepath)
    print('\nDownloaded', statinfo.st_size, 'bytes.')

def _remove_tarballs(tarball_url_list, db_dir):
    """Helper function that will remove the splitted tarballs

    Args:
        tarball_url_list: A file containing the tarball name, URL, SHA256 hash
        db_dir: The directory where the tarballs are
    """
    with open(tarball_url_list) as f:
        for line in f.readlines():
            tarball, _, _ = line.split()
            tarball_path = os.path.join(db_dir, tarball)
            if os.path.isfile(tarball_path):
                os.remove(tarball_path)

def _extract_plantas50(tarball_url_list, db_dir):
    """Helper function will extract the Plantas50.tar

    Args:
        tarball_url_list: A file containing the tarball name, URL, SHA256 hash
        db_dir: The directory where the tarballs are
    """
    print("Plantas50.tar is consistent. Extracting...")
    _remove_tarballs(tarball_url_list, db_dir)
    tarball_path = os.path.join(db_dir, 'Plantas50.tar')
    tarball = tarfile.open(tarball_path, 'r:')
    tarball.extractall(db_dir)
    os.remove(tarball_path)
    print("Plantas50 extracted. Ready!")

def _extract_tar_files_list(fileslistpath, db_dir):
    """Helper function will extract the download_file_list.tar.bz2

    Args:
        fileslistpath: Path to download_file_list.tar.bz2
        db_dir: The directory where the tarballs are
    """
    tarball = tarfile.open(fileslistpath, 'r:bz2')
    tarball.extractall(db_dir)
    os.remove(fileslistpath)

def download_and_merge(db_dir):
    """This function will download all tarballs and merge them

    Args:
        db_dir: The directory that the dataset will be saved
    """
    # Download download_file_list.tar.bz2
    fileslistpath = os.path.join(db_dir, 'download_file_list.tar.bz2')
    if os.path.isfile(fileslistpath):
        print("download_file_list.tar.bz2 already downloaded. Checking consistency...")
        if _check_hash_sha256(fileslistpath, DLF_SHA256):
            _extract_tar_files_list(fileslistpath, db_dir)
        else:
            print("download_file_list.tar.bz2 is corrupted. Re-downloading...")
            os.remove(fileslistpath)
            _download(DLF_URL, fileslistpath)
            _extract_tar_files_list(fileslistpath, db_dir)
    else:
        print("download_file_list.tar.bz2 does not exists. Downloading...")
        _download(DLF_URL, fileslistpath)
        _extract_tar_files_list(fileslistpath, db_dir)
    tarball_url_list = os.path.join(db_dir, 'tar_url_hash.txt')

    # Verify if Plantas50 exists and its consistency
    file = os.path.join(db_dir, 'Plantas50.tar')
    if os.path.isfile(file):
        print("Plantas50.tar already downloaded. Checking consistency...")
        if _check_hash_sha256(file, P50_SHA256):
            _extract_plantas50(tarball_url_list, db_dir)
            if os.path.isfile(tarball_url_list):
                os.remove(tarball_url_list)
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
                    _download(url, tarball_path)
                    print("Checking consistency...")
                    assert (_check_hash_sha256(tarball_path, sha256)), \
                    "Downloaded file is corrupted. \
                    Try again later or contact owner"
            else:
                print("Downloading " + tarball + " ...")
                _download(url, tarball_path)
                print("Checking consistency...")
                assert (_check_hash_sha256(tarball_path, sha256)), \
                "Downloaded file is corrupted. Try again later or contact owner"

    # Merge tarballs
    print("Merging tarballs into Plantas50.tar. This should take awhile...")
    tarballs_path = os.path.join(db_dir, 'Plantas50.tar.part')

    if os.name == 'nt':
        os.system("copy /b " + tarballs_path + "* " + file)
    else:
        os.system("cat " + tarballs_path + "* > " + file)

    if _check_hash_sha256(file, P50_SHA256):
        _extract_plantas50(tarball_url_list, db_dir)
    else:
        print("Plantas50.tar is corrupted. Re-run the script.")
        os.remove(file)

    if os.path.isfile(tarball_url_list):
        os.remove(tarball_url_list)

def main(argv):
    db_dir = os.path.abspath('./')

    if len(argv) == 2:
        db_dir = os.path.abspath(argv[1])
        assert (os.path.isdir(db_dir)), "Not a valid directory."
    else:
        print("The command should be python download_plantas50.py \
            dataset_directory")
        return

    download_and_merge(db_dir)

if __name__ == "__main__":
    main(sys.argv)
