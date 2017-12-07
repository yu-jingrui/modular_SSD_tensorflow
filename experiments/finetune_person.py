# Continue experiment step 3


from trainer.train_params import TrainerParams
from evaluator.eval_params import EvaluatorParams


# Fine tune from 184587 steps.
step3_1 = TrainerParams(
    feature_extractor='vgg_16',
    model_name='ssd512',
    fine_tune_fe=True,
    train_dir='../experiments/ssd512_voc0712_29-11-2017/logs/finetune_person',
    checkpoint_path='../experiments/ssd512_voc0712_29-11-2017/logs/finetune/model.ckpt-184587',
    ignore_missing_vars=False,
    learning_rate=0.001,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=1,
    num_epochs_per_decay=1,
    end_learning_rate=0.01,
    max_number_of_steps=230000,
    optimizer='adam',
    weight_decay=0.0005,
    batch_size=10,
    log_every_n_steps=100,
    save_interval_secs=60*60,
    save_summaries_secs=60,
    labels_offset=0,
    matched_thresholds=0.5
    )