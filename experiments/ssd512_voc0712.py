# Train VGG_16 SSD512 on VOC data, 29.11.2017
# 1. Step: train on voc0712 trainval, without vertical flipping and rotation
# 2. Step: validate on voc07 test
# 3. Step: fine tune on voc07 person + HDA + PIROPO with vertical flipping and rotation
# 4. Step: validate on HDA + PIROPO

from trainer.train_params import TrainerParams
from evaluator.eval_params import EvaluatorParams

# -------------------------------------------------------- #
# Train VGG16-SSD512 on VOC0712 Trainval
step1_1 = TrainerParams(
    feature_extractor='vgg_16',
    model_name='ssd512',
    fine_tune_fe=False,
    train_dir='../experiments/ssd512_voc0712_29-11-2017/logs',
    checkpoint_path='../checkpoints/vgg_16.ckpt',
    ignore_missing_vars=False,
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
    save_summaries_secs=30,
    labels_offset=0,
    matched_thresholds=0.5
    )
# At 18719 steps, total_loss=~4.3
# After 18719 steps, reduce batch_size to 16 because global_step fluctuates. Also, make room for evaluation.
# Original batch_size by @LevinJ is 32, so raise max_number_of_steps to 55k.
# Also reduce per_process_gpu_memory_fraction to 0.8 to allow some room for other applications and reduce OOM problems.
step1_2 = TrainerParams(
    feature_extractor='vgg_16',
    model_name='ssd512',
    fine_tune_fe=False,
    train_dir='../experiments/ssd512_voc0712_29-11-2017/logs',
    checkpoint_path='../checkpoints/vgg_16.ckpt',
    ignore_missing_vars=False,
    learning_rate=0.1,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=1,
    num_epochs_per_decay=1,
    end_learning_rate=0.1,
    max_number_of_steps=55000,
    optimizer='adam',
    weight_decay=0.0005,
    batch_size=16,
    log_every_n_steps=100,
    save_interval_secs=60*60,
    save_summaries_secs=30,
    labels_offset=0,
    matched_thresholds=0.5
    )


# -------------------------------------------------------- #
# Evaluate when not training
eval1 = EvaluatorParams(
    checkpoint_path='/home/yjin/SSD/experiments/ssd512_voc0712_29-11-2017/logs/',
    use_finetune=False,
    is_training=False,
    eval_train_dataset=False,
    loop=True,
    which_checkpoint=None
    )
# At 18719 steps, mAP on VOC07test is 0.464
