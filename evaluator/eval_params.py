from collections import namedtuple


EvaluatorParams = namedtuple(
    'EvaluatorParameters',
    ['checkpoint_path',  # directory must have '/' at the end
     'eval_dir',  # directory to save evaluation results
     'use_finetune',  # whether use checkpoints under 'finetune/' folder
     'is_training',  # whether evaluate while training is ongoing
     'eval_train_dataset',  # whether evaluate against training dataset
     'loop',  # whether evaluate in loops
     'which_checkpoint'  # specify a checkpoint to evaluate
     ])


eval_while_training = EvaluatorParams(
    checkpoint_path='./logs/',
    eval_dir='./logs/eval',
    use_finetune=False,
    is_training=True,
    eval_train_dataset=False,
    loop=True,
    which_checkpoint=None
    )

eval_while_finetuning = EvaluatorParams(
    checkpoint_path='./logs/',
    eval_dir='./logs/eval',
    use_finetune=True,
    is_training=True,
    eval_train_dataset=False,
    loop=True,
    which_checkpoint=None
)

eval_only_last_ckpt = EvaluatorParams(
    checkpoint_path='./logs/',
    eval_dir='./logs/eval',
    use_finetune=False,
    is_training=False,
    eval_train_dataset=False,
    loop=False,
    which_checkpoint='last'
)

eval_only_last_ckpt_finetune = EvaluatorParams(
    checkpoint_path='./logs/',
    eval_dir='./logs/eval',
    use_finetune=True,
    is_training=False,
    eval_train_dataset=False,
    loop=False,
    which_checkpoint='last'
)

debug_params = EvaluatorParams(
    checkpoint_path='/home/yjin/SSD/tmp/logs/',
    eval_dir='./logs/eval',
    use_finetune=False,
    is_training=False,
    eval_train_dataset=True,
    loop=False,
    which_checkpoint='last'
)
