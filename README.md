# Modularized SSD implementation in TensorFlow

This repo tries to implement SSD in a modularized fashion.

Inspiration: Speed/accuracy trade-offs for modern convolutional object detectors. (arXiv:1611.10012)


## Dependencies:
- Python 3.x
- Tensorflow 1.x
- CUDA 8.0
- OpenCV 3.x

## HOWTO:
#### Prepare data for training
#### Train SSD layers
#### Fine tune feature extractor
#### Get training and test results
#### Demo 

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

