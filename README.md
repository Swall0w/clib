clib
====

[![Build Status](https://travis-ci.org/Swall0w/clib.svg?branch=master)](https://travis-ci.org/Swall0w/clib)
[![Code Climate](https://codeclimate.com/github/Swall0w/clib/badges/gpa.svg)](https://codeclimate.com/github/Swall0w/clib)
[![Issue Count](https://codeclimate.com/github/Swall0w/clib/badges/issue_count.svg)](https://codeclimate.com/github/Swall0w/clib)
[![Test Coverage](https://codeclimate.com/github/Swall0w/clib/badges/coverage.svg)](https://codeclimate.com/github/Swall0w/clib/coverage)
[![codecov](https://codecov.io/gh/Swall0w/clib/branch/master/graph/badge.svg)](https://codecov.io/gh/Swall0w/clib)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

A helpful library for modeling neural networks

## Description

`clib` is modeling Neural Networks tool that contains common function for training.
In `clib`, there're functions and classes implemented with chainer.
In `examples`, there're train and test scripts.
We use Trainer for training codes.

## Requirement

* numpy
* chainer
* cv2
* python3
* PIL

## Usage

train and test script are in examples. See them.

## Installation

1. Clone this repository.
2. Build
    ```console
    $ make build
    ```
2. Install
    ```console
    $ make install
    ```
After installation, you only test train code in examples.

## Features

* Extentions of slack notifications
* SOMs
* VAE
* ConvolutinalAutoEncoder
* AlexNet model
* Links of convLSTM
* Darknet19
* Yolov2

## Todo

* Eliminate opencv in clib except train script.
* Testcode

## Future Plan
### Image Recognition

* NIN model
* Densely Connected Convolutional Networks (very similar to RoR?)
* ResNet model
* Deep Networks with Stochastic Depth
* FractalNet
* Residual Networks of Residual Networks: Multilevel Residual Networks (RoR)
* Aggregated Residual Transformations for Deep Neural Networks

### Object Detection

* SSD
* DSSD model
* Feature Pyramid Networks

### Image Generative Model
* pix2pix model
* Cycle Gan model
* Disco Gan model
* LSGan model
* Pixel Recurrent Neural Networks
* StackGAN
* Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning
* Generating Videos with Scene Dynamics

### Image Segmentation 

* SigNET model
* [FlowNet](https://arxiv.org/abs/1504.06852)
* [FlowNet2](https://github.com/lmb-freiburg/flownet2)

### Sound Processing

* [WaveNet](https://github.com/musyoku/wavenet)
* SoundNet

### NLP

* Pointer Sentinel Mixture Models
* LightRNN
* Google's Neural Machine Translation system: Bridging the Gap between Human and Machine Translation
* Word2Vec
* Fast TEXT

### Else

* Hybrid computing using a neural network with dynamic external memory
* Learning to learn by gradient descent by gradient descent
* Convolutional Seq2Seq model
* [Seq2Seq](http://qiita.com/kenchin110100/items/b34f5106d5a211f4c004)

### Features

* CNN visualizer
* Extentions of Visualization on Django?
* Network Visualizer with [viz.js?](https://github.com/mdaines/viz.js)

## Infomation

* [2016年のディープラーニング論文100選](http://qiita.com/sakaiakira/items/9da1edda802c4884865c)
* [ディープラーニング関連の○○Netまとめ](http://qiita.com/shinya7y/items/8911856125a3109378d6#_reference-a60de5539cc2a2dd8bd7)
* [DeepLearning研究 2016年のまとめ](http://qiita.com/eve_yk/items/f4b274da7042cba1ba76)
* [2016年の深層学習を用いた画像認識モデル](http://qiita.com/aiskoaskosd/items/59c49f2e2a6d76d62798)
* [chainer](https://github.com/chainer/chainer/wiki/External-examples)
