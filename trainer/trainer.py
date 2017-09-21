from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as tf_saver

from preparedata import PrepareData
from nets.ssd import g_ssd_model
from postprocessingdata import g_post_processing_data


class Trainer(PrepareData):
    def __init__(self):
        PrepareData.__init__(self)
        self.num_epochs_per_decay = 2.0
        self.learning_rate_decay_type = 'exponential'
        self.end_learning_rate = 0.0001
        self.learning_rate = 0.01

        # Optimizer
        self.optimizer = 'rmsprop'

        self.adadelta_rho = 0.95
        self.opt_epsilon = 1.0
        self.adagrad_initial_accumulator_value = 0.1
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.ftrl_learning_rate_power = -0.5
        self.ftrl_initial_accumulator_value = 0.1
        self.ftrl_l1 = 0.0
        self.ftrl_l2 = 0.0
        self.momentum = 0.9

        self.rmsprop_decay = 0.9
        self.rmsprop_momentum = 0.9

        self.train_dir = '/tmp/tfmodel/'
        self.max_number_of_steps = None

        self.checkpoint_path = None
        self.checkpoint_exclude_scopes = None
        self.ignore_missing_vars = False

        self.batch_size = 32

        self.save_interval_secs = 60 * 60  # one hour
        self.save_summaries_secs = 60

        self.label_smoothing = 0

