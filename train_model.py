from collections import namedtuple

from ssd.ssdmodel import SSDModel
from trainer.trainer import Trainer
from trainer.preparedata import PrepareData
from trainer.postprocessingdata import PostProcessingData


ssd_model = SSDModel(feature_extractor='vgg_16', model_name='ssd512')
data_preparer = PrepareData(ssd_model, batch_size=32, labels_offset=0, matched_thresholds=0.5)
data_postprocessor = PostProcessingData(ssd_model)
ssd_trainer = Trainer(ssd_model, data_preparer, data_postprocessor)

if __name__ == '__main__':
    ssd_trainer.start_training()
