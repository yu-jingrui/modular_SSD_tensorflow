from evaluator.evaluator import EvaluatorParams
from trainer.trainer import TrainerParams

# -------------------------------------------------------- #
# Training parameters for MobileNet-SSD512
train1_1 = TrainerParams(
    feature_extractor='mobilenet_v1',
    model_name='ssd512',
    fine_tune_fe=False,
    train_dir='../experiments/mobilenet_11-12-2017/logs',
    checkpoint_path='../checkpoints/mobilenet/mobilenet_v1_1.0_224.ckpt',
    ignore_missing_vars=False,
    learning_rate=0.1,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=1,
    num_epochs_per_decay=1,
    end_learning_rate=0.1,
    max_number_of_steps=30000,
    optimizer='adam',
    weight_decay=0.0005,
    batch_size=24,
    log_every_n_steps=20,
    save_interval_secs=60*60,
    save_summaries_secs=60,
    labels_offset=0,
    matched_thresholds=0.5
    )
