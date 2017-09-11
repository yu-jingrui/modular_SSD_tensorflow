import math
from collections import namedtuple
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim
from tensorflow.python.ops import array_ops

import tf_extended as tfe
from utils import custom_layers
from ssd import ssd_common

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

ssd300_params = SSDParams(model_name='ssd300',
                          img_shape=(300, 300),
                          num_classes=21,
                          no_annotation_label=21,
                          feature_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
                          feature_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
                          anchor_size_bounds=[0.15, 0.90],
                          anchor_sizes=[(21., 45.),
                                        (45., 99.),
                                        (99., 153.),
                                        (153., 207.),
                                        (207., 261.),
                                        (261., 315.)],
                          anchor_ratios=[[2, .5],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5],
                                         [2, .5]],
                          anchor_steps=[8, 16, 32, 64, 100, 300],
                          anchor_offset=0.5,
                          normalizations=[20, -1, -1, -1, -1, -1],
                          prior_scaling=[0.1, 0.1, 0.2, 0.2]
                          )

ssd512_params = SSDParams(model_name='ssd512',
                          img_shape=(512, 512),
                          num_classes=21,
                          no_annotation_label=21,
                          feature_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12'],
                          feature_shapes=[(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)],
                          anchor_size_bounds=[0.10, 0.90],
                          anchor_sizes=[(20.48, 51.2),
                                        (51.2, 133.12),
                                        (133.12, 215.04),
                                        (215.04, 296.96),
                                        (296.96, 378.88),
                                        (378.88, 460.8),
                                        (460.8, 542.72)],
                          anchor_ratios=[[2, .5],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5],
                                         [2, .5]],
                          anchor_steps=[8, 16, 32, 64, 128, 256, 512],
                          anchor_offset=0.5,
                          normalizations=[20, -1, -1, -1, -1, -1, -1],
                          prior_scaling=[0.1, 0.1, 0.2, 0.2]
                          )


class SSD_Model(model_name):
    def __init__(self):
        if model_name == 'ssd300':
            self.params = ssd300_params
        elif model_name == 'ssd512':
            self.params = ssd512_params

    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME', data_format=data_format):
                with slim.arg_scope([custom_layers.pad2d,
                                     custom_layers.l2_normalization,
                                     custom_layers.channel_to_last],
                                    data_format=data_format) as scope:
                    return scope

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0,
               negative_ratio=3.,
               alpha=1,
               scope=None):
        with tf.name_scope(scope, 'ssd_losses'):
            lshape = tfe.get_shape(logits[0], 5)
            num_classes = lshape[-1]
            # batch_size = lshape[0]

            # Flatten out all vectors
            flogits = []
            fgclasses = []
            fgscores = []
            flocalisations = []
            fglocalisations = []
            for i in range(len(logits)):
                flogits.append(tf.reshape(logits[i], [-1, num_classes]))
                fgclasses.append(tf.reshape(gclasses[i], [-1]))
                fgscores.append(tf.reshape(gscores[i], [-1]))
                flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
                fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
            # And concat the crap!
            logits = tf.concat(flogits, axis=0)
            gclasses = tf.concat(fgclasses, axis=0)
            gscores = tf.concat(fgscores, axis=0)
            localisations = tf.concat(flocalisations, axis=0)
            glocalisations = tf.concat(fglocalisations, axis=0)
            dtype = logits.dtype

            # Compute positive matching mask...
            pmask = gscores > match_threshold
            fpmask = tf.cast(pmask, dtype)
            n_positives = tf.reduce_sum(fpmask)

            # Hard negative mining...
            # for no_classes, we only care that false positive's label is 0
            # this is why pmask suffice our needs
            no_classes = tf.cast(pmask, tf.int32)
            predictions = slim.softmax(logits)
            nmask = tf.logical_not(pmask)
            fnmask = tf.cast(nmask, dtype)
            nvalues = tf.where(nmask, predictions[:, 0], 1. - fnmask)
            nvalues_flat = tf.reshape(nvalues, [-1])

            # Number of negative entries to select.
            max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
            n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
            n_neg = tf.minimum(n_neg, max_neg_entries)
            # avoid n_neg is zero, and cause error when doing top_k later on
            n_neg = tf.maximum(n_neg, 1)

            val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
            max_hard_pred = -val[-1]
            # Final negative mask, hard negative mining
            nmask = tf.logical_and(nmask, nvalues <= max_hard_pred)
            fnmask = tf.cast(nmask, dtype)

            # Add cross-entropy loss.
            with tf.name_scope('cross_entropy_pos'):
                total_cross_pos = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=gclasses)
                total_cross_pos = tf.reduce_sum(total_cross_pos * fpmask, name="cross_entropy_pos")
                tf.losses.add_loss(total_cross_pos)

            with tf.name_scope('cross_entropy_neg'):
                total_cross_neg = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=no_classes)
                total_cross_neg = tf.reduce_sum(total_cross_neg * fnmask, name="cross_entropy_neg")
                tf.losses.add_loss(total_cross_neg)

            # Add localization loss: smooth L1, L2, ...
            with tf.name_scope('localization'):
                # Weights Tensor: positive mask + random negative.
                weights = tf.expand_dims(alpha * fpmask, axis=-1)
                total_loc = custom_layers.abs_smooth_2(localisations - glocalisations)
                total_loc = tf.reduce_sum(total_loc * weights, name="localization")
                tf.losses.add_loss(total_loc)

            total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')

            # Add to EXTRA LOSSES TF.collection
            tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
            tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
            tf.add_to_collection('EXTRA_LOSSES', total_cross)
            tf.add_to_collection('EXTRA_LOSSES', total_loc)

            # stick with the original paper in terms of defining model loss
            model_loss = tf.get_collection(tf.GraphKeys.LOSSES)
            model_loss = tf.add_n(model_loss)
            model_loss = array_ops.where(tf.equal(n_positives, 0),
                                         array_ops.zeros_like(model_loss),
                                         tf.div(1.0, n_positives) * model_loss)
            # Add regularization loss
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            regularization_loss = tf.add_n(regularization_losses, name='regularization_loss')

            # if model loss is zero, no need to do gradient update on this batch
            total_loss = array_ops.where(tf.equal(n_positives, 0),
                                         array_ops.zeros_like(model_loss),
                                         tf.add(model_loss, regularization_loss))

            # debugging info
            tf.summary.scalar("postive_num", n_positives)
            tf.summary.scalar("negative_num", n_neg)
            tf.summary.scalar("regularization_loss", regularization_loss)
            # with tf.name_scope('variables_loc'):
            #     selected_p = tf.boolean_mask(glocalisations, pmask)
            #     p_mean, p_variance = tf.nn.moments(selected_p, [0])
            #     tf.summary.scalar("mean_cx", p_mean[0])
            #     tf.summary.scalar("mean_cy", p_mean[1])
            #     tf.summary.scalar("mean_w", p_mean[2])
            #     tf.summary.scalar("mean_h", p_mean[3])
            #
            #     tf.summary.scalar("var_cx", p_variance[0])
            #     tf.summary.scalar("var_cy", p_variance[1])
            #     tf.summary.scalar("var_w", p_variance[2])
            #     tf.summary.scalar("var_h", p_variance[3])

            return total_loss
