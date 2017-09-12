import tensorflow as tf
from tensorflow.python.ops import array_ops

import tf_extended as tfe
from utils import custom_layers
from ssd import ssd_common

slim = tf.contrib.slim


def get_losses(logits, localisations, gclasses, glocalisations, gscores,
               match_threshold=0, negative_ratio=2.5, alpha=1., scope=None):
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

        # stick with the original paper in terms of definig model loss
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


def detected_bboxes(predictions, localisations, num_classes,
                    select_threshold=0.01, nms_threshold=0.45,
                    clipping_bbox=None, top_k=400, keep_top_k=200):
    """Get the detected bounding boxes from the SSD network output.
    """
    # Select top_k bboxes from predictions, and clip
    rscores, rbboxes = \
        ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                        select_threshold=select_threshold,
                                        num_classes=num_classes)
    rscores, rbboxes = \
        tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
    # Apply NMS algorithm.
    rscores, rbboxes = \
        tfe.bboxes_nms_batch(rscores, rbboxes,
                             nms_threshold=nms_threshold,
                             keep_top_k=keep_top_k)
    if clipping_bbox is not None:
        rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
    return rscores, rbboxes