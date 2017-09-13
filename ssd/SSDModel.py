import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np
import math

import tf_extended as tfe
from utils import custom_layers
from ssd import ssd_utils
from ssd import ssd_blocks
from nets import nets_factory as nf

slim = tf.contrib.slim


class SSDModel:
    """
    Implementation of the SSD network.
    """

    # ============================= PUBLIC METHODS ============================== #
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
            self.params = ssd_blocks.ssd300_params
            self.__ssd_blocks = ssd_blocks.ssd300
        else:
            self.params = ssd_blocks.ssd512_params
            self.__ssd_blocks = ssd_blocks.ssd512
        self.__feature_extractor = nf.get_network_fn(feature_extractor, self.params.num_classes)

    # TODO: construct the whole SSD network
    def get_model(self, inputs, weight_decay=0.0005, is_training=False):
        """

        :param inputs:
        :param weight_decay:
        :param is_training:
        :return:
        """
        return

    def get_anchors_all_layers(self, dtype=np.float32):
        """Compute anchor boxes for all feature layers.
        """
        # ssd_anchors_all_layers()
        layers_anchors = []
        for i, s in enumerate(self.params.feature_shapes):
            anchor_bboxes = ssd_utils.anchor_one_layer(self.params.img_shape, s,
                                                       self.params.anchor_sizes[i],
                                                       self.params.anchor_ratios[i],
                                                       self.params.anchor_steps[i],
                                                       offset=self.params.anchor_offset,
                                                       dtype=dtype)
            layers_anchors.append(anchor_bboxes)
        return layers_anchors

    def decode_bboxes_layer(self, feat_localizations, anchors):
        """convert ssd boxes from relative to input image anchors to relative to
        input width/height, for one single feature layer

        Return:
          numpy array Batches x H x W x 4: ymin, xmin, ymax, xmax
        """

        l_shape = feat_localizations.shape
        #         if feat_localizations.shape != anchors.shape:
        #             raise "feat_localizations and anchors should be of identical shape, and corresond to each other"

        # Reshape for easier broadcasting.
        feat_localizations = feat_localizations[np.newaxis, :]
        anchors = anchors[np.newaxis, :]

        xref = anchors[..., 0]
        yref = anchors[..., 1]
        wref = anchors[..., 2]
        href = anchors[..., 3]

        # Compute center, height and width
        cy = feat_localizations[..., 1] * href * self.params.prior_scaling[0] + yref
        cx = feat_localizations[..., 0] * wref * self.params.prior_scaling[1] + xref
        h = href * np.exp(feat_localizations[..., 3] * self.params.prior_scaling[2])
        w = wref * np.exp(feat_localizations[..., 2] * self.params.prior_scaling[3])

        # bboxes: ymin, xmin, xmax, ymax.
        bboxes = np.zeros_like(feat_localizations)
        bboxes[..., 0] = cy - h / 2.
        bboxes[..., 1] = cx - w / 2.
        bboxes[..., 2] = cy + h / 2.
        bboxes[..., 3] = cx + w / 2.
        bboxes = np.reshape(bboxes, l_shape)
        return bboxes

    def decode_bboxes_all_layers(self, localizations):
        """convert ssd boxes from relative to input image anchors to relative to
        input width/height

        Return:
          numpy array Batches x H x W x 4: ymin, xmin, ymax, xmax
        """
        decoded_bboxes = []
        all_anchors = self.get_allanchors()
        for i in range(len(localizations)):
            decoded_bboxes.append(self.decode_bboxes_layer(localizations[i], all_anchors[i]))

        return decoded_bboxes

    # ============================= PRIVATE METHODS ============================= #