import tensorflow as tf
from utils import custom_layers
slim = tf.contrib.slim


def ssd500(net, end_points):
    """
    Implementation of the SSD512 network.

    No prediction and localization layers included!!!

    """

    # Block 6: 3x3 conv
    net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
    net = slim.batch_norm(net)
    end_points['block6'] = net

    # Block 7: 1x1 conv
    net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
    net = slim.batch_norm(net)
    end_points['block7'] = net

    # Block 8/9/10/11/12: 1x1 and 3x3 convolutions stride 2 (except last).
    end_point = 'block8'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
    end_points[end_point] = net

    end_point = 'block9'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
    end_points[end_point] = net

    end_point = 'block10'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
    end_points[end_point] = net

    end_point = 'block11'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
    end_points[end_point] = net

    end_point = 'block12'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 256, [4, 4], scope='conv4x4', padding='VALID')
        net = slim.batch_norm(net)
    end_points[end_point] = net

    return end_points
