from ssd.ssdmodel import SSDModel
from trainer.trainer import Trainer
from trainer.preparedata import PrepareData
from trainer.postprocessingdata import PostProcessingData
from trainer import train_params


# basic parameters for training
feature_extractor = 'vgg_16'
model_name = 'ssd512'
params_for_training = train_params.ssd_train_params  # trainer parameters
batch_size = params_for_training.batch_size
labels_offset = 0
matched_thresholds = 0.5


if __name__ == '__main__':
    ssd_model = SSDModel(feature_extractor, model_name)
    data_preparer = PrepareData(ssd_model, batch_size, labels_offset, matched_thresholds)
    data_postprocessor = PostProcessingData(ssd_model)
    ssd_trainer = Trainer(ssd_model, data_preparer, data_postprocessor, train_params.ssd_train_params)
    ssd_trainer.start_training()
