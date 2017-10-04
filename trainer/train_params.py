from collections import namedtuple


TrainerParams = namedtuple('TrainerParameters',
                           ['fine_tune_fe',  # (True / False) whether feature extractor should be fine tuned
                            'train_dir',  # directory to save model weights
                            'checkpoint_path',  # directory of the initial / pre trained weights
                            'ignore_missing_vars',  # (True / False) whether to ignore layers in the pre trained weights
                            'learning_rate',  # initial learning rate
                            'learning_rate_decay_type',
                            'learning_rate_decay_factor',
                            'num_epochs_per_decay',
                            'end_learning_rate',
                            'max_number_of_steps',  # maximal number of training steps
                            'optimizer',
                            'weight_decay',
                            'batch_size',
                            'log_every_n_steps',
                            'save_interval_secs',
                            'save_summaries_secs'
                            ])


ssd_train_params = TrainerParams(fine_tune_fe=False,
                                 train_dir='./logs',
                                 checkpoint_path='./checkpoints/vgg_16.ckpt',
                                 ignore_missing_vars=True,
                                 learning_rate=0.1,
                                 learning_rate_decay_type='fixed',
                                 learning_rate_decay_factor=1,
                                 num_epochs_per_decay=1,
                                 end_learning_rate=0.1,
                                 max_number_of_steps=30000,
                                 optimizer='adam',
                                 weight_decay=0.0005,
                                 batch_size=20,
                                 log_every_n_steps=100,
                                 save_interval_secs=60*60,
                                 save_summaries_secs=60
                                 )


ssd_finetune_params1 = TrainerParams(fine_tune_fe=True,
                                     train_dir='./logs/finetune',
                                     checkpoint_path='./logs',
                                     ignore_missing_vars=False,
                                     learning_rate=0.01,
                                     learning_rate_decay_type='fixed',
                                     learning_rate_decay_factor=None,
                                     num_epochs_per_decay=None,
                                     end_learning_rate=None,
                                     max_number_of_steps=90000,
                                     optimizer='adam',
                                     weight_decay=0.0005,
                                     batch_size=32,
                                     log_every_n_steps=100,
                                     save_interval_secs=60*60,
                                     save_summaries_secs=60
                                     )