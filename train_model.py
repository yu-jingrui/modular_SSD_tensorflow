from ssd.ssdmodel import SSDModel
from trainer.trainer import Trainer
from trainer.preparedata import PrepareData
from trainer.postprocessingdata import PostProcessingData
from trainer import train_params


if __name__ == '__main__':
    params = train_params.ssd_train_params

    feature_extractor = params.feature_extractor
    model_name = params.model_name
    batch_size = params.batch_size
    labels_offset = params.labels_offset
    matched_thresholds = params.matched_thresholds

    ssd_model = SSDModel(feature_extractor, model_name)
    data_preparer = PrepareData(ssd_model, batch_size, labels_offset, matched_thresholds)
    data_postprocessor = PostProcessingData(ssd_model)
    ssd_trainer = Trainer(ssd_model, data_preparer, data_postprocessor, params)

    ssd_trainer.start_training()
