import os
import os.path
import fileinput
import shutil
import math

import argparse
import time
import progressbar

from google.protobuf import text_format
import numpy as np
import PIL.Image
import scipy.misc

from sklearn import svm
from sklearn.externals import joblib

os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output
import caffe
from caffe.proto import caffe_pb2

def get_net(caffemodel, deploy_file, use_gpu=True):
    """
    Returns an instance of caffe.Net

    Arguments:
    caffemodel -- path to a .caffemodel file
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    if use_gpu:
        caffe.set_mode_gpu()

    # load a new model
    return caffe.Net(deploy_file, caffemodel, caffe.TEST)

def get_transformer(deploy_file, mean_file=None):
    """
    Returns an instance of caffe.io.Transformer

    Arguments:
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(
            inputs = {'data': dims}
            )
    t.set_transpose('data', (2,0,1)) # transpose to (channels, height, width)

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2,1,0))

    if mean_file:
        # set mean pixel
        with open(mean_file,'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            t.set_mean('data', pixel)

    return t

def load_image(path, height, width, mode='RGB'):
    """
    Load an image from disk

    Returns an np.ndarray (channels x width x height)

    Arguments:
    path -- path to an image on disk
    width -- resize dimension
    height -- resize dimension

    Keyword arguments:
    mode -- the PIL mode that the image should be converted to
        (RGB for color or L for grayscale)
    """
    image = PIL.Image.open(path)
    image = image.convert(mode)
    image = np.array(image)
    # squash
    image = scipy.misc.imresize(image, (height, width), 'bilinear')
    return image

def forward_pass(images, net, transformer, layer, batch_size=None):
    """
    Returns features for each image as an np.ndarray (nImages x nFeatures)

    Arguments:
    images -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer

    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    if batch_size is None:
        batch_size = 1

    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:,:,np.newaxis])
        else:
            caffe_images.append(image)

    dims = transformer.inputs['data'][1:]

    scores = None
    for chunk in [caffe_images[x:x+batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        output = net.forward(blobs=[layer])[layer]
        if scores is None:
            scores = np.copy(output)
        else:
            scores = np.vstack((scores, output))

    return scores

def read_labels(labels_file):
    """
    Returns a list of strings

    Arguments:
    labels_file -- path to a .txt file
    """
    if not labels_file:
        print 'WARNING: No labels file provided. Results will be difficult to interpret.'
        return None

    labels = []
    with open(labels_file) as infile:
        for line in infile:
            label = line.strip()
            if label:
                labels.append(label)
    assert len(labels), 'No labels found'
    return labels

def get_descriptors(caffemodel, arch_path, set_path, nets_batch_size, networks,
        img_size, layers, use_gpu=True):
    """
    Get the last layer before classification descriptors
    """

    # Set paths
    trainset_file = os.path.join(set_path, 'train.txt')
    testset_file = os.path.join(set_path, 'test.txt')
    mean_file = os.path.join(set_path, 'mean.binaryproto')
    labels_file = os.path.join(set_path, 'labels.txt')

    # Read Images
    image_files = {'train': [], 'test': []}
    true_labels = {'train': [], 'test': []}
    with open(trainset_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            image_files['train'].append(' '.join(line[:-1]))
            true_labels['train'].append(int(line[-1]))

    with open(testset_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            image_files['test'].append(' '.join(line[:-1]))
            true_labels['test'].append(int(line[-1]))

    # Test
    # image_files['train'] = image_files['train'][:350]
    # image_files['test'] = image_files['test'][:80]
    # true_labels['train'] = true_labels['train'][:350]
    # true_labels['test'] = true_labels['test'][:80]

    batch_size_reading = 32

    for setn in ['train', 'test']:
        for arch, derivs in networks.iteritems():
            for deriv in derivs:
                print('Processing: ' + setn + '/' + arch + '-' + deriv)

                # Load the model
                model_path = os.path.join(arch_path, arch + '-' + deriv)
                deploy_file = os.path.join(model_path, 'deploy.prototxt')

                net = get_net(caffemodel[arch], deploy_file, use_gpu)
                transformer = get_transformer(deploy_file, mean_file)

                # Get the descriptors for each image
                descriptors = None
                num_images = len(image_files[setn])
                i = 0
                chunks = [image_files[setn][x:x+batch_size_reading] for x in xrange(0, num_images, batch_size_reading)]

                pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()], max_value=len(chunks)).start()
                for chunk in chunks:
                    #print('Reading chunk: ' + str(i + 1) + '/' + str(len(chunks)))
                    images = [load_image(image_file, img_size, img_size, 'RGB') for image_file in chunk]
                    new_desc = forward_pass(images, net, transformer, layers[arch], batch_size=int(nets_batch_size[arch]))
                    if arch == 'GoogLeNet':
                        new_desc = new_desc[:, :, 0, 0]
                    if descriptors is None:
                        descriptors = new_desc
                    else:
                        descriptors = np.vstack((descriptors, new_desc))
                    pbar.update(i + 1)
                    i += 1
                pbar.finish()

                np.save(os.path.join(model_path, setn + '-descriptors.npy'), np.array(descriptors))
                np.save(os.path.join(model_path, 'true-label-' + setn + '.npy'), np.array(true_labels[setn]))

def train_SVM(arch_path, networks):
    """
    Train a SVM with the descriptors

    """

    for arch, derivs in networks.iteritems():
        for deriv in derivs:
            print('Training SVM: ' + arch + '-' + deriv)
            model_path = os.path.join(arch_path, arch + '-' + deriv)

            # Read descriptors and labels
            true_labels = np.load(os.path.join(model_path, 'true-label-train.npy'))
            descriptors = np.load(os.path.join(model_path, 'train-descriptors.npy'))

            # Train SVM
            lin_clf = svm.LinearSVC()
            lin_clf.fit(descriptors, true_labels)

            joblib.dump(lin_clf, os.path.join(model_path,'trained_svm.pkl'))

def classify_and_save(arch_path, networks):

    for arch, derivs in networks.iteritems():
        for deriv in derivs:
            print('Classifying: ' + arch + '-' + deriv)
            model_path = os.path.join(arch_path, arch + '-' + deriv)

            # Read descriptors and labels
            true_labels = np.load(os.path.join(model_path, 'true-label-test.npy'))
            descriptors = np.load(os.path.join(model_path, 'test-descriptors.npy'))

            # Read SVM
            lin_clf = joblib.load(os.path.join(model_path,'trained_svm.pkl'))

            # Get scores
            scores = lin_clf.decision_function(descriptors)

            # Get Top-5
            indices = (-scores).argsort()[:, :5] # take top 5 results

            # Get accuracy
            top1 = 0.0
            top5 = 0.0
            for image_index, index_list in enumerate(indices):
                if true_labels[image_index] == index_list[0]:
                    top1 += 1.0
                if true_labels[image_index] in index_list:
                    top5 += 1.0

            print('Top-1 Accuracy: ' + str(top1 / len(true_labels) * 100.0) + '%')
            print('Top-5 Accuracy: ' + str(top5 / len(true_labels) * 100.0) + '%')

            # Save List
            np.save(os.path.join(model_path, 'prediction-top-5-test.npy'), np.array(indices))
            np.save(os.path.join(model_path, 'prediction-score-test.npy'), np.array(scores))


networks = {'CaffeNet': ['SVM'], 'GoogLeNet': ['SVM']}

core_dataset = '/usr/share/digits/digits/jobs/20160326-172626-a652'
full_dataset = '/usr/share/digits/digits/jobs/20160326-143203-b20e'
archictetures_core_path = '/home/reneoctavio/Documents/Plantas/Training/Core'
archictetures_full_path = '/home/reneoctavio/Documents/Plantas/Training/Full'


nets_batch_size = {'AlexNet': 100,
            'CaffeNet': 256,
            'GoogLeNet': 32,
            'Inception': 12,
            'ResNet-50': 6,
            'ResNet-101': 4}

finetune_files = {'CaffeNet': '/home/reneoctavio/Downloads/bvlc_reference_caffenet.caffemodel',
                    'GoogLeNet': '/home/reneoctavio/Downloads/bvlc_googlenet.caffemodel',
                    'ResNet-50': '/home/reneoctavio/Downloads/ResNet-50-model.caffemodel'}

img_size = 224

layers = {'CaffeNet': 'fc7', 'GoogLeNet': 'pool5/7x7_s1'}

# Core
get_descriptors(finetune_files, archictetures_core_path, core_dataset, nets_batch_size, networks, img_size, layers)
train_SVM(archictetures_core_path, networks)
classify_and_save(archictetures_core_path, networks)

# Full
get_descriptors(finetune_files, archictetures_full_path, full_dataset, nets_batch_size, networks, img_size, layers)
train_SVM(archictetures_full_path, networks)
classify_and_save(archictetures_full_path, networks)
