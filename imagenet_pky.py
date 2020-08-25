
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
dataType = args.data
model_type = args.type
model_spec = 'r50_1x_sk0'
# model_type = 'supervised'
# model_spec = 'r50_1x_sk0'
# dataType = 'train'  # datatype (train / val)
dataHome = '/SSD_data/Imagenet2012'  # Imagenet2012 path (classification)
dataPath = os.path.join(dataHome, dataType)
print('dataPath:', dataPath)

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
data_total = 50000
batch_size = 100
num_batch = data_total // batch_size
# device
device = '/gpu:0'

savePath = '/home/user/Desktop/pky/simclr/save/Imagenet/'  # save path
timestr = time.strftime("%m%d-%H%M")  # time stamp
print('savePath:', savePath)
print('timeStamp:', timestr)

print('load hub modules...')
# model_type = 'supervised'
# model_spec = 'r50_1x_sk0'
model_name = model_type+'_'+model_spec
hub_path = 'gs://simclr-checkpoints/simclrv2/'+model_type+'/'+model_spec+'/hub/'
# hub_path = 'gs://simclr-checkpoints/simclrv2/pretrained/r50_1x_sk0/hub/'  # self-supervised
# hub_path = 'gs://simclr-checkpoints/simclrv2/finetuned_1pct/r50_1x_sk0/hub/'  # 1% fine-tuned
# hub_path = 'gs://simclr-checkpoints/simclrv2/finetuned_10pct/r50_1x_sk0/hub/' # 10% fine-tuned
# hub_path = 'gs://simclr-checkpoints/simclrv2/finetuned_100pct/r50_1x_sk0/hub/' # 100% fine-tuned
# hub_path = 'gs://simclr-checkpoints/simclrv2/supervised/r50_1x_sk0/hub/'  # supervised
module = hub.Module(hub_path, trainable=False)
print('load finished!')


def _load_imagenet():
    imagenet_data = torchvision.datasets.ImageNet(
        dataPath, split=dataType, download=False, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        imagenet_data, batch_size=batch_size, shuffle=True, num_workers=10, worker_init_fn=_init_fn)
    for n_batch, (data, label) in enumerate(data_loader):
        if n_batch >= num_batch:
            return
        print("batch {}/{} is loaded!".format(n_batch+1, num_batch))
        yield (data.numpy()).transpose(0, 2, 3, 1), label.numpy()


with tf.device("/gpu:0"):
    input_tensor = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    keys = module(inputs=input_tensor, signature="default", as_dict=True)
    features = keys['default']
    logits = keys['logits_sup']

print("initializing sessions...")
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# sess.run(tf.compat.v1.global_variables_initializer())
print("initializing finished!")

features_total = np.empty((0, 2048))
logits_total = np.empty((0, 1000))
labels_total = np.array([])

print("batch computation start...")
for batch_data, batch_label in _load_imagenet():
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
