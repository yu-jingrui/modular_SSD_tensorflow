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
#### Train only SSD layers
Download pre trained weights for the backend first. See projects in acknowledgement for reference.

Specify the training parameters in *trainer/train_params.py*. Pay attention to format, because if parameters are not completely defined, there will be ERROR. *ssd_params_train* is the reference. Then set the params to use in *train_model.py* and run it. Start TensorBoard and you can monitor the training process.
#### Fine tune feature extractor
Much like training the SSD layers, just define a set of params and specify it in *train_model.py* and run. *ssd_finetune_params1* and *ssd_finetune_params2* are references which conforms with the guideline given in [SSD_tensorflow_VOC](https://github.com/LevinJ/SSD_tensorflow_VOC). My personal experience is, the model still improves if the training steps are extended.
#### Get training and test results
NOT YET POSSIBLE. Test routine is not yet implemented.
#### Demo 
Just run *new_demo.py*. All parameters need are defined inside this file. Pay attention to conformity.

## Workflow for implementing a new backend for SSD
**1. Prepare base network**  
- A base network is the backend (i.e. feature extractor) for the SSD net, like vgg_16. Therefore `vgg_16_base()` in _nets/vgg.py_ serves as a good example. The network body should only contain convolutional layers for the feature extraction and no fc layers. The base network function should have the parameter `inputs` and return `nets, end_points`. In the body of the function, a variable scope (i.e. name of the network) should be defined. 
- Define the `arg_scope` of the network in a separate function, like `vgg_base_arg_scope()`. For now only `weigth_decay` and `data_format='NHWC'` should be used as parameters.
- If you are defining **your own network**, the returned `net` should have the size [H, W] = [64, 64] for SSD512, or [38, 38] for SSD300 in the last two layers/blocks. If you are adapting an existing net, see **4**.

**2. Add base network in _nets/nets_factory.py_**  
Add the file in imports and and the keys for the base network and its arg_scope in `base_networks_map` and `base_arg_scopes_map`

**3. Define a training parameter set**  
Define it in _trainer/train_params_. Set `fine_tune_fe=True` if there is no pre trained weights to use.

**4. Find the feature layer**  
Set breakpoint in `get_model()` in _ssd/ssdmodel.py_ (see comment for more info). Set the training parameters in _./train_model.py_ and run _train_model.py_ in debug mode. On the breakpoint, by reading into values of `end_points` find out the feature layer to use (according to SSD paper, the output of the second last layer/block should be used). Also make sure you have the right tensor shape at this step. Add an item in `feature_layer` in _ssd/ssd_blocks.py_ for your base network.

**5. Train your model**  
If you got everything right, you can now train your network with customized backend. If you adopted an existing base network which was pre trained on ImageNet, follow the training routine defined for other networks in _train_params.py_. If you have defined a new base network, set `fine_tune_fe=True` and `checkpoint_path=''` and train.

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
