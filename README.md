# Modularized SSD implementation in TensorFlow

This repo tries to implement SSD in a modularized fashion.

Inspiration: Speed/accuracy trade-offs for modern convolutional object detectors. (arXiv:1611.10012)

## Dependencies:
- Python 3.x
- Tensorflow 1.x
- CUDA 8.0
- OpenCV 3.x

## Current state
VGG16-SSD300, VGG16-SSD512, MobileNet-SSD300, MobileNet-SSD512 are trainable. For SSD300, batch_size=32 is safe with 8GB of GRAM. VGG16-SSD512 can do 20 and MobileNet-SSD512 can do 16. For fine tuning batch size must be reduced (halved would be OK).

## HOWTO:
#### Prepare data for training
See projects in Acknowledgment. They have documented this quite well. Scripts needed are still not included in this repo.
#### Train SSD layers
Specify the training parameters in *trainer/train_params.py*. Pay attention to format. If parameters are not completely defined, there will be ERROR. *ssd_params_train* is the reference. Then set the params to use in *train_model.py* and run it. Start TensorBoard and you can monitor the training process.
#### Fine tune feature extractor
Much like training the SSD layers, just define a set of params and specify it in *train_model.py* and run. *ssd_finetune_params1* and *ssd_finetune_params2* are references which conforms with the guideline given in [SSD_tensorflow_VOC](https://github.com/LevinJ/SSD_tensorflow_VOC). My personal experience is, the model still improves if the training steps are extended.
#### Get training and test results
NOT YET POSSIBLE. Test routine is not yet implemented.
#### Demo 
Just run *new_demo.py*. All parameters need are defined inside this file. Pay attention to conformity.
## TODOs:
- [x] Rewrite core Tensorflow SSD for modularization
- [x] Connect original VGG backend for SSD
- [ ] Implement test routine
- [ ] Train and test VGG-SSD
- [ ] Connect other backends for SSD
- [ ] Train and test connected backends


## Acknowledgement
This repo is based on the works:
* [SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow/): The Tensorflow re-implementation of Caffe SSD
* [SSD_tensorflow_VOC](https://github.com/LevinJ/SSD_tensorflow_VOC): A fork of SSD-Tensorflow with bachnorm and simplified structure
#### Why I used SSD_tensorflow_VOC instead of SSD-Tensorflow as baseline
Because I am still a rookie and it is just so much easier to understand and modify. Besides I find the way to define parameters for training in SSD-Tensorflow is hard to track. The get lost in the training outputs so quickly. Or it is just because I am such a rookie. :stuck_out_tongue:
