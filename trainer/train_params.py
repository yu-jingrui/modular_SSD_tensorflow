from collections import namedtuple


TrainingParams = namedtuple('Training Parameters',
                            [
                                'train_dir',
                                'checkpoint_path',
                                'checkpoint_exlude_scope',
                                'ignore_missing_vars'
                                'max_num_of_steps',
                                'log_every_n_steps'
                                'batch_size',
                                'learning_rate',
                                'learning_rate_decay_type',
                                'end_learning_rate',
                                'num_epochs_per_decay',
                                'trainable_scopes'
                                'optimizer',
                                'adadelta_rho',
                                'opt_epsilon',
                                'adagrad_initial_accumulator_value',
                                'adam_beta1',
                                'adam_beta2',
                                'ftrl_learning_rate_power',
                                'ftrl_initial_accumulatro_value'
                                'ftrl_l1',
                                'ftrl_l2',
                                'momentum',
                                'rmsprop_decay',
                                'rmsprop_momentum',
                                'label_smoothing',
                                'save_summaries_secs',
                                'save_interval_secs'
                            ])