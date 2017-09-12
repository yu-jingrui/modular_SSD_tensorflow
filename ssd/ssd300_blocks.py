import tensorflow as tf
from utils import custom_layers
slim = tf.contrib.slim


def ssd300(net, end_points):
    """
    Implementation of the SSD300 network.

    The default features layers with 300x300 image input are:
      conv4 ==> 38 x 38
      conv7 ==> 19 x 19
      conv8 ==> 10 x 10
      conv9 ==> 5 x 5
      conv10 ==> 3 x 3
      conv11 ==> 1 x 1
    The default image size used to train this network is 300x300.

    No prediction and localization layers included!!!
    """
    # block 6: 3x3 conv
    net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
    net = slim.batch_norm(net)
    net = custom_layers.dropout_with_noise(net)
    end_points['block6'] = net

    # block 7: 1x1 conv
    net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
    net = slim.batch_norm(net)
    net = custom_layers.dropout_with_noise(net)
    end_points['block7'] = net

    # block 8/9/10/11: 1x1 and 3x3 convolutions with stride 2 (except lasts)
    end_point = 'block8'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.dropout_with_noise(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
        net = custom_layers.dropout_with_noise(net)
    end_points[end_point] = net

    end_point = 'block9'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.dropout_with_noise(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 256, stride=2, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
        net = custom_layers.dropout_with_noise(net)
    end_points[end_point] = net

    end_point = 'block10'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.dropout_with_noise(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 256, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
        net = custom_layers.dropout_with_noise(net)
    end_points[end_point] = net

    end_point = 'block11'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.dropout_with_noise(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 256, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
        net = custom_layers.dropout_with_noise(net)
    end_points[end_point] = net

    return net, end_points
