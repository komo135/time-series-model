# Time series model using tensorflow
## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Install](#install)

## General info
This repository contains models that have been converted from image models and layers to time series.

## Technologies
- python 3.7 ~ 3.10
- tensorflow 2.7.0
- numpy 1.21.4

## Install
install package
```console
cd time-series-model-main
python setup.py
```
### example
```python
import tftime

model = tftime.build_model("sam_efficientnet_b2", (30, 1), 2, None)
```
```python
import tftime
import tensorflow as tf

inputs = tf.keras.layers.Input((30, 1))

x = tftime.block.ConvBlock(128, "conv1d", "resnet")(inputs)
x = tf.keras.layers.GlobalAvgPool1D()(x)
x = tftime.layers.Output(5, "softmax")

model = tftime.model.SAMModel(inputs, x)
model.compile("adam", categorical_crossentropy, ["accuracy"])

model.fit(x, y)
```

### available model
```python
print(tftime.available_network)
```
```
['efficientnet b0 ~ b8' 'sam_efficientnet b0 ~ b8'
 'dense_efficientnet b0 ~ b8' 'sam_dense_efficientnet b0 ~ b8'
 'lambda_efficientnet b0 ~ b8' 'sam_lambda_efficientnet b0 ~ b8'
 'efficientnetv2 b0 ~ b8' 'sam_efficientnetv2 b0 ~ b8' 'resnet b0 ~ b8'
 'sam_resnet b0 ~ b8' 'se_resnet b0 ~ b8' 'sam_se_resnet b0 ~ b8'
 'densenet b0 ~ b8' 'sam_densenet b0 ~ b8' 'se_densenet b0 ~ b8'
 'sam_se_densenet b0 ~ b8' 'lambda_resnet b0 ~ b8'
 'sam_lambda_resnet b0 ~ b8' 'se_lambda_resnet b0 ~ b8'
 'sam_se_lambda_resnet b0 ~ b8' 'convnext b0 ~ b8' 'sam_convnext b0 ~ b8'
 'se_convnext b0 ~ b8' 'sam_se_convnext b0 ~ b8' 'lambda_convnext b0 ~ b8'
 'sam_lambda_convnext b0 ~ b8' 'se_lambda_convnext b0 ~ b8'
 'sam_se_lambda_convnext b0 ~ b8']
```
