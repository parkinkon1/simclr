
import argparse
import time
import torchvision
import torch
from torchvision import transforms as T
from PIL import Image
import importlib.util

import tensorflow_datasets as tfds
import tensorflow_hub as hub
import sys
import os
import yaml
import re
import numpy as np
import subprocess
import random

# import tensorflow.compat.v1 as tf
import tensorflow as tf
# tf.disable_eager_execution()

# subprocess.check_call(["gcloud", "auth", "application-default", "login"])
seed = 201711075
torch.manual_seed(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)


def _init_fn(worker_id):
    seed_s = seed + worker_id
    np.random.seed(seed_s)
    random.seed(seed_s)
    torch.manual_seed(seed_s)
    return


parser = argparse.ArgumentParser(description='feature extraction')
parser.add_argument('-d', '--data', required=True,
                    type=str, help='dataType(train/val)')
parser.add_argument('-t', '--type', required=True, type=str,
                    help='modelType(pretrained/supervised)')
args = parser.parse_args()


# 1-3. Load data
dataType = args.data  # datatype (train / test)
model_type = args.type  # supervised / pretrained
model_spec = 'r50_1x_sk0'
dataPath = '/home/user/Desktop/pky/simclr/save/Imagenet/dataset/'
print('dataPath:', dataPath)

data_total = 50000
batch_size = 100
num_batch = data_total // batch_size
# device
device = '/gpu:0'

savePath = '/home/user/Desktop/pky/simclr/save/STL10/'  # save path
timestr = time.strftime("%m%d-%H%M")  # time stamp
print('savePath:', savePath)
print('timeStamp:', timestr)

print('load hub modules...')
# model_type = 'supervised'
# model_spec = 'r50_1x_sk0'
model_name = model_type+'_'+model_spec
hub_path = 'gs://simclr-checkpoints/simclrv2/'+model_type+'/'+model_spec+'/hub/'
module = hub.Module(hub_path, trainable=False)
print('load finished!')


file_total = 5


def _load_stl10(prefix="train"):
    transform = T.Compose([T.Resize(224), T.ToTensor()])
    data = np.fromfile('./stl10_binary/' + prefix +
                       '_X.bin', dtype=np.uint8)
    label = np.fromfile('./stl10_binary/' + prefix +
                        '_y.bin', dtype=np.uint8)
    data = (transform(data).numpy()).transpose(0, 2, 3, 1)
    label = label - 1

    print("{} images".format(prefix))
    print(data.shape, label.shape)
    for i in range(0, data.shape[0], batch_size):
        yield data[i: i+batch_size], label[i: i+batch_size]


# def _load_imagenet():
#     for file_num in range(file_total):
#         data = np.load(dataPath+dataType+'_images_{}.npy'.format(file_num))
#         label = np.load(dataPath+dataType+'_labels_{}.npy'.format(file_num))
#         for i in range(0, data.shape[0], batch_size):
#             yield data[i: i+batch_size], label[i: i+batch_size]


# with tf.device("/gpu:0"):
#     input_tensor = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
#     keys = module(inputs=input_tensor, signature="default", as_dict=True)
#     features = keys['default']
#     logits = keys['logits_sup']

input_tensor = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
keys = module(inputs=input_tensor, signature="default", as_dict=True)
features = keys['default']
logits = keys['logits_sup']

print("initializing sessions...")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("initializing finished!")

features_total = np.empty((0, 2048))
logits_total = np.empty((0, 1000))
labels_total = np.array([])

print("batch computation start...")
for batch_data, batch_label in _load_stl10(dataType):
    features_, logits_ = sess.run((features, logits), feed_dict={
        input_tensor: batch_data})
    features_total = np.append(features_total, features_, axis=0)
    logits_total = np.append(logits_total, logits_, axis=0)
    labels_total = np.append(labels_total, batch_label, axis=0)
print("batch Finished!")

# 저장
np.save(savePath+timestr+'_'+dataType+'_features_'+model_name, features_total)
np.save(savePath+timestr+'_'+dataType+'_logits_'+model_name, logits_total)
np.save(savePath+timestr+'_'+dataType+'_labels_'+model_name, labels_total)

print("features:", str(features_total.shape))
print("logits:", str(logits_total.shape))
print("labels:", str(labels_total.shape))
