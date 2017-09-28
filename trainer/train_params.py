from collections import namedtuple


reference_params = {'num_epochs_per_decay': 2.0,
                    'learning_rate_decay_type': 'exponential',
                    'learning_rate_decay_factor': 0.1,
                    'end_learning_rate': 0.0001,
                    'learning_rate': 0.01,

                    'optimizer': 'rmsprop',
                    'weight_decay': 0.0005,
                    'adadelta_rho': 0.95,
                    'opt_epsilon': 1.0,
                    'adagrad_initial_accumulator_value': 0.1,
                    'adam_beta1': 0.9,
                    'adam_beta2': 0.999,
                    'ftrl_learning_rate_power': -0.5,
                    'ftrl_initial_accumulator_value': 0.1,
                    'ftrl_l1': 0.0,
                    'ftrl_l2': 0.0,
                    'momentum': 0.9,
                    'rmsprop_decay': 0.9,
                    'rmsprop_momentum': 0.9,
                    'train_dir': './logs',
                    'max_number_of_steps': None,
                    'log_every_n_steps': None,
                    'checkpoint_path': None,
                    'checkpoint_exclude_scopes': None,
                    'ignore_missing_vars': False,
                    'batch_size': 32,
                    'save_interval_secs': 60 * 60,  # one hour
                    'save_summaries_secs': 60,
                    'label_smoothing': 0,
                    'fine_tune_fe': True
                   }


TrainerParames = namedtuple('TrainerParameters', ['fine_tune_fe',
                                                  'train_dir',
                                                  'checkpoint_path',
                                                  'checkpoint_exlude_scopes',
                                                  'trainable_scopes',
                                                  'learning_rate',
                                                  'learning_rate_decay_type',
                                                  'max_number_of_steps',
                                                  'optimizer',
                                                  'weight_decay',
                                                  'batch_size',
                                                  'log_every_n_steps',
                                                  'save_interval_secs',
                                                  'save_summaries_secs',
                                                  'label_smoothing',
                                                  'ignore_missing_vars'
                                                  ])

ssd_train_params = TrainerParames(fine_tune_fe=False,
                                  train_dir='./logs',
                                  checkpoint_path='./checkpoints/vgg_16.ckpt',
                                  checkpoint_exlude_scopes=g_ssd.model_name,
                                  trainable_scopes=g_ssd.model_name,
                                  learning_rate=0.1,
                                  learning_rate_decay_type='fixed',
                                  max_number_of_steps=30000,
                                  optimizer='adam',
                                  weight_decay=0.0005,
                                  batch_size=32,
                                  log_every_n_steps=100,
                                  save_interval_secs=60*60,
                                  save_summaries_secs=60,
                                  label_smoothing=0,
                                  ignore_missing_vars=False
                                 )