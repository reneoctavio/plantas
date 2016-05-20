# Imports
import os
import os.path
import time
import numpy as np
import PIL.Image
import scipy.misc
import caffe
import pandas as pd

from google.protobuf import text_format
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

def forward_pass(images, net, transformer, batch_size=None):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)

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
        start = time.time()
        output = net.forward()[net.outputs[-1]]
        end = time.time()
        if scores is None:
            scores = np.copy(output)
        else:
            scores = np.vstack((scores, output))
        #print 'Processed %s/%s images in %f seconds ...' % (len(scores), len(caffe_images), (end - start))

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

def classify(caffemodel, deploy_file, test_db_path,
        mean_file=None, labels_file=None, batch_size=None, use_gpu=True, sensitivity=5):
    """
    Classify some images against a Caffe model and print the results

    Arguments:
    caffemodel -- path to a .caffemodel
    deploy_file -- path to a .prototxt
    image_files -- list of paths to images and true label

    Keyword arguments:
    mean_file -- path to a .binaryproto
    labels_file path to a .txt file
    use_gpu -- if True, run inference on the GPU
    """
    # Load the model and images
    net = get_net(caffemodel, deploy_file, use_gpu)
    transformer = get_transformer(deploy_file, mean_file)
    _, channels, height, width = transformer.inputs['data']
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)


    image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(test_db_path) for f in fn if f.endswith((".jpg", ".JPG", "*.jpeg", "*.JPEG"))]

    labels = read_labels(labels_file)

    true_labels = []

    for img_file in image_files:
        file_path = os.path.dirname(img_file)
        label = file_path.split(os.sep)[-1]
        true_labels.append(labels.index(label))

    # Batch images for reading
    batch_size_reading = batch_size
    num_images = len(image_files)
    num_calc_images = 0
    true_imgs = 0
    true_files = []

    # Calculate time
    start_total = time.time()

    for chunk in [(image_files[x:x+batch_size_reading], true_labels[x:x+batch_size_reading]) \
                  for x in xrange(0, num_images, batch_size_reading)]:

        # Calculate time
        start = time.time()

        # Load images
        images = [load_image(image_file, height, width, mode) \
                  for image_file in chunk[0]]

        # Classify the image
        scores = forward_pass(images, net, transformer, batch_size=batch_size)

        ### Process the results
        indices = (-scores).argsort()[:, :sensitivity] # take top sensitivity
        for image_index, index_label in enumerate(indices):
            if (chunk[1][image_index] in index_label):
                true_files.append(chunk[0][image_index])
                true_imgs += 1

        num_calc_images += len(chunk[1])

        print('Files calculated:   ' + str(num_calc_images) + ' of ' + str(num_images))
        print('Correct images:     ' + str(true_imgs) + ' of ' + str(num_calc_images))

        print('Time elapsed batch: ' + str(time.time() - start))

        # Backup after every 50 batches
        if (num_calc_images % (50 * batch_size_reading) == 0):
            df = pd.DataFrame({'Files': true_files})
            df.to_csv(TEST_DB_PATH + 'true_files.csv')
            print('Saved!')

    print('Time elapsed      : ' + str((time.time() - start_total) / 60))

    df = pd.DataFrame({'Files': true_files})
    df.to_csv(TEST_DB_PATH + 'true_files.csv')
    print('Saved!')


os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output

# Set paths
MODEL_PATH    = '/usr/share/digits/digits/jobs/20160302-132337-4388/'
DATASET_PATH  = '/usr/share/digits/digits/jobs/20160301-201544-5755/'
SNAPSHOT_PATH = '/usr/share/digits/digits/jobs/20160302-132337-4388/'
OUTPUT_PATH   = '/home/reneoctavio/Documents/Plantas/Caffe/googlenet_default_finetuned_prelu/'
TEST_DB_PATH  = '/home/reneoctavio/Documents/Plantas/New/'

#TEST_DB_PATH  = '/media/reneoctavio/221CF0C91CF098CD/Users/reneo/Downloads/Extreme Picture Finder'

# Input Files
MODEL_FILE    = MODEL_PATH    + 'deploy.prototxt'
PRETRAINED    = SNAPSHOT_PATH + 'snapshot_iter_4560.caffemodel'
LABEL_FILE    = DATASET_PATH  + 'labels.txt'
MEAN_FILE     = DATASET_PATH  + 'mean.binaryproto'

classify(PRETRAINED, MODEL_FILE, TEST_DB_PATH, mean_file=MEAN_FILE, labels_file=LABEL_FILE, batch_size=64, sensitivity=1)
