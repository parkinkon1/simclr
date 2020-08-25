

from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.decomposition import PCA
import argparse
import time
import torchvision
import torch
from torchvision import transforms as T
from PIL import Image
import importlib.util

import sys
import os
import yaml
import re
import numpy as np
import subprocess
import random

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
args = parser.parse_args()


# 1-3. Load data
dataType = args.data
dataHome = '/SSD_data/Imagenet2012'  # Imagenet2012 path (classification)
dataPath = os.path.join(dataHome, dataType)
print('dataPath:', dataPath)

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
data_total = 50000
batch_size = 100
num_batch = data_total // batch_size


savePath = '/home/user/Desktop/pky/simclr/save/Imagenet/'  # save path
timestr = time.strftime("%m%d-%H%M")  # time stamp
print('savePath:', savePath)
print('timeStamp:', timestr)


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


images_total = np.empty((0, 224, 224, 3))
labels_total = np.array([])

print("batch computation start...")
for batch_image, batch_label in _load_imagenet():
    images_total = np.append(images_total, batch_image, axis=0)
    labels_total = np.append(labels_total, batch_label, axis=0)
print("batch Finished!")
print(images_total.shape, labels_total.shape)

np.save(savePath+dataType+'_images', images_total)
np.save(savePath+dataType+'_labels', labels_total)


print("TSNE: transform start...")
tsne = TSNE(n_components=2)
images_tsne = tsne.fit_transform(images_total)
print("TSNE: transform finished!")

# 저장
np.save(savePath+dataType+'_tsne_images', images_tsne)
print("TSNE: file saved")
