import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type = int, default = 0, help = 'GPU to use [default: GPU 0]')
parser.add_argument('--model', default = 'pointnet_cls',
                    help = 'Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type = int, default = 5, help = 'example size to vis [default: 5]')
parser.add_argument('--num_point', type = int, default = 1024,
                    help = 'Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default = 'log1/model.ckpt',
                    help = 'model checkpoint file path [default: log1/model.ckpt]')
parser.add_argument('--vis_mode', default = 'critical', help = 'Vis mode: critical[default]/all')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODE = FLAGS.vis_mode
MODEL = importlib.import_module(FLAGS.model)  # import network module

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
               open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))]

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
        os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))


def get_sample(BATCH_SIZE):
    current_data, current_label = provider.loadDataFile(TRAIN_FILES[0])
    current_data = current_data[:, 0:NUM_POINT, :]
    current_label = np.squeeze(current_label)
    print(current_data.shape)

    points = current_data[0:BATCH_SIZE]
    labels = current_label[0:BATCH_SIZE]
    print labels
    return points, labels


def get_all_points():
    '''
    We choose points in the inscribed cube of unit sphere as our input space. We compute 80
    points each axis.
    '''
    lim = 0.577
    all_points = []

    for x in np.linspace(-lim, lim, 80):
        for y in np.linspace(-lim, lim, 80):
            for z in np.linspace(-lim, lim, 80):
                all_points.append([x, y, z])

    all_points = np.array(all_points)
    all_points = np.reshape(all_points, (500, 1024, 3))
    num = all_points.shape[0]
    all_label = np.zeros(num)
    return all_points, all_label


def get_vis_file(BATCH_SIZE, points, label, MODE):
    is_training = False
    with tf.device('/gpu:' + str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape = ())

        # simple model
        _, _, hx, maxpool = MODEL.get_model(pointclouds_pl, is_training_pl)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config = config)
    saver.restore(sess, MODEL_PATH)
    print("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'hx': hx,
           'maxpool': maxpool}
    is_training = False
    feed_dict = {ops['pointclouds_pl']: points,
                 ops['labels_pl']: label,
                 ops['is_training_pl']: is_training}
    hx, maxpool = sess.run([ops['hx'], ops['maxpool']], feed_dict = feed_dict)
    print hx.shape, maxpool.shape
    np.savez('{}.npz'.format(MODE), points = points, hx = hx, maxpool = maxpool)


if __name__ == '__main__':
    with tf.Graph().as_default():
        sample_points, sample_label = get_sample(BATCH_SIZE)
        all_points, all_label = get_all_points()
        if (MODE == 'critical'):
            get_vis_file(BATCH_SIZE, sample_points, sample_label, 'critical')
        if (MODE == 'all'):
            get_vis_file(500, all_points, all_label, 'all')
        else:
            print 'please input the right vis mode, type -h for more information'
