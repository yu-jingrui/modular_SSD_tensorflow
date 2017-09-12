import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.python.ops import array_ops
import tf_extended as tfe
import numpy as np
import math

from utils import custom_layers
from ssd import ssd_common
from ssd import ssd300_blocks
from ssd import ssd512_blocks
from nets import nets_factory as nf


class SSDModel:
    """
    Implementation of the SSD network.
    """
    def __init__(self, feature_extractor, model_name):
        """
        Initialize an instance of the SSDModel
        :param feature_extractor: name of the feature extractor (backbone)
        :param model_name: name of the SSD model to use: ssd300 or ssd500
        """
        if feature_extractor not in nf.networks_map:
            raise ValueError('Feature extractor unknown: %s.', feature_extractor)
        if model_name not in ['ssd300', 'ssd512']:
            raise ValueError('Choose model between ssd300 and ssd512.')

        if model_name == 'ssd300':
            self.params = ssd300_blocks.params
            self.__ssd_blocks = ssd300_blocks.ssd300
        else:
            self.params = ssd512_blocks.params
            self.__ssd_blocks = ssd512_blocks.ssd512
        self.__feature_extractor = nf.get_network_fn(feature_extractor, self.params.num_classes)

    def get_model(self, inputs, weight_decay=0.0005, is_training=False):
        """

        :param inputs:
        :param weight_decay:
        :param is_training:
        :return:
        """
        return

    def get_losses(self, logits, localisations,
                   gclasses, glocalisations, gscores,
                   match_threshold=0,
                   negative_ratio=2.5,
                   alpha=1.,
                   scope=None):
        """Loss functions for training the SSD 300 VGG network.

        This function defines the different loss components of the SSD, and
        adds them to the TF loss collection.

        Arguments:
          logits: (list of) predictions logits Tensors;
          localisations: (list of) localisations Tensors;
          gclasses: (list of) groundtruth labels Tensors;
          glocalisations: (list of) groundtruth localisations Tensors;
          gscores: (list of) groundtruth score Tensors;
        """
        with tf.name_scope(scope, 'ssd_losses'):
            lshape = tfe.get_shape(logits[0], 5)
            num_classes = lshape[-1]
            # batch_size = lshape[0]

            # Flatten out all vectors!
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
            pmask = gclasses > match_threshold
            fpmask = tf.cast(pmask, dtype)
            n_positives = tf.reduce_sum(fpmask)

            # Hard negative mining...
            # for no_classes, we only care that false positive's label is 0
            # this is why pmask sufice our needs
            no_classes = tf.cast(pmask, tf.int32)
            predictions = slim.softmax(logits)
            nmask = tf.logical_not(pmask)

            fnmask = tf.cast(nmask, dtype)
            nvalues = tf.where(nmask,
                               predictions[:, 0],
                               1. - fnmask)
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
                total_cross_pos = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                                 labels=gclasses)
                total_cross_pos = tf.reduce_sum(
                    total_cross_pos * fpmask, name="cross_entropy_pos")
                tf.losses.add_loss(total_cross_pos)

            with tf.name_scope('cross_entropy_neg'):
                total_cross_neg = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                                 labels=no_classes)
                total_cross_neg = tf.reduce_sum(
                    total_cross_neg * fnmask, name="cross_entropy_neg")
                tf.losses.add_loss(total_cross_neg)

            # Add localization loss: smooth L1, L2, ...
            with tf.name_scope('localization'):
                # Weights Tensor: positive mask + random negative.
                weights = tf.expand_dims(alpha * fpmask, axis=-1)
                total_loc = custom_layers.abs_smooth_2(
                    localisations - glocalisations)
                total_loc = tf.reduce_sum(
                    total_loc * weights, name="localization")
                tf.losses.add_loss(total_loc)

            total_cross = tf.add(
                total_cross_pos, total_cross_neg, 'cross_entropy')

            # Add to EXTRA LOSSES TF.collection
            tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
            tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
            tf.add_to_collection('EXTRA_LOSSES', total_cross)
            tf.add_to_collection('EXTRA_LOSSES', total_loc)

            # stick with the orgiginal paper in terms of definig model loss
            model_loss = tf.get_collection(tf.GraphKeys.LOSSES)
            model_loss = tf.add_n(model_loss)
            model_loss = array_ops.where(tf.equal(n_positives, 0), array_ops.zeros_like(model_loss),
                                         tf.div(1.0, n_positives) * model_loss)
            # Add regularziaton loss
            regularization_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)
            regularization_loss = tf.add_n(
                regularization_losses, name='regularization_loss')

            # if model oss is zero, no need to do gradient update on this batch
            total_loss = array_ops.where(tf.equal(n_positives, 0), array_ops.zeros_like(model_loss),
                                         tf.add(model_loss, regularization_loss))

            # debugging info
            tf.summary.scalar("postive_num", n_positives)
            tf.summary.scalar("negative_num", n_neg)
            tf.summary.scalar("regularization_loss", regularization_loss)
            #             with tf.name_scope('variables_loc'):
            #                 selected_p = tf.boolean_mask(glocalisations, pmask)
            #                 p_mean, p_variance = tf.nn.moments(selected_p, [0])
            #                 tf.summary.scalar("mean_cx", p_mean[0])
            #                 tf.summary.scalar("mean_cy", p_mean[1])
            #                 tf.summary.scalar("mean_w", p_mean[2])
            #                 tf.summary.scalar("mean_h", p_mean[3])
            #
            #                 tf.summary.scalar("var_cx", p_variance[0])
            #                 tf.summary.scalar("var_cy", p_variance[1])
            #                 tf.summary.scalar("var_w", p_variance[2])
            #                 tf.summary.scalar("var_h", p_variance[3])

            return total_loss