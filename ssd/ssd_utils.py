"""
This file contains methods and functions used by SSDModel.
These are not directly needed for training and evaluation.
"""

import tensorflow as tf
import numpy as np
import math
from collections import namedtuple
from utils import custom_layers
from ssd import ssd_utils

slim = tf.contrib.slim


# =========================================================================== #
# Definition of the parameter structure
# =========================================================================== #
SSDParams = namedtuple('SSDParameters', ['model_name',
                                         'img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feature_layers',
                                         'feature_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'])


def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d for s, d in zip(static_shape, dynamic_shape)]


def multibox_layer(inputs, num_classes, anchor_sizes, anchor_ratios, normalization):
    """
    Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True)

    # Number of anchors.
    num_anchors = len(anchor_sizes) + len(anchor_ratios)

    # Location.
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None, scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred, ssd_utils.tensor_shape(loc_pred, 4)[:-1] + [num_anchors, 4])

    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None, scope='conv_cls')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred, ssd_utils.tensor_shape(cls_pred, 4)[:-1] + [num_anchors, num_classes])

    return cls_pred, loc_pred


def compute_jaccard(gt_bboxes, anchors):

    gt_bboxes = tf.reshape(gt_bboxes, (-1, 1, 4))
    anchors = tf.reshape(anchors, (1, -1, 4))

    inter_ymin = tf.maximum(gt_bboxes[:, :, 0], anchors[:, :, 0])
    inter_xmin = tf.maximum(gt_bboxes[:, :, 1], anchors[:, :, 1])
    inter_ymax = tf.minimum(gt_bboxes[:, :, 2], anchors[:, :, 2])
    inter_xmax = tf.minimum(gt_bboxes[:, :, 3], anchors[:, :, 3])

    h = tf.maximum(inter_ymax - inter_ymin, 0.)
    w = tf.maximum(inter_xmax - inter_xmin, 0.)

    inter_area = h * w
    anchors_area = (anchors[:, :, 3] - anchors[:, :, 1]) * (anchors[:, :, 2] - anchors[:, :, 0])
    gt_bboxes_area = (gt_bboxes[:, :, 3] - gt_bboxes[:, :, 1]) * (gt_bboxes[:, :, 2] - gt_bboxes[:, :, 0])
    union_area = anchors_area - inter_area + gt_bboxes_area
    jaccard = inter_area / union_area

    return jaccard


def anchor_one_layer(img_shape, feat_shape, sizes, ratios, step, offset=0.5, dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors,), dtype=dtype)
    w = np.zeros((num_anchors,), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i + di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i + di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w
