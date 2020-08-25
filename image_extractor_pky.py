

# from sklearn.manifold import TSNE
# from sklearn import preprocessing
# from sklearn.decomposition import PCA
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


batch_size = 100

num_classes = 1000
data_per_class = 50
data_total = num_classes * data_per_class
num_batch = data_total // batch_size

class_count = np.zeros(num_classes)


savePath = '/home/user/Desktop/pky/simclr/save/Imagenet/dataset/'  # save path
print('savePath:', savePath)


images_total = np.empty((0, 224, 224, 3))
labels_total = np.array([])

data_per_files = 10000
file_count = 0


def _save_data(data, label, finished=False):
    global images_total, labels_total, data_per_files, file_count

    if finished:
        np.save(savePath+dataType +
                '_images_{}'.format(file_count), images_total)
        np.save(savePath+dataType +
                '_labels_{}'.format(file_count), labels_total)
        return

    images_total = np.append(images_total, data, axis=0)
    labels_total = np.append(labels_total, label, axis=0)
    if images_total.shape[0] > data_per_files:
        np.save(savePath+dataType +
                '_images_{}'.format(file_count), images_total)
        np.save(savePath+dataType +
                '_labels_{}'.format(file_count), labels_total)
        images_total = np.empty((0, 224, 224, 3))
        labels_total = np.array([])
        file_count += 1


def _load_imagenet(save=False):
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    imagenet_data = torchvision.datasets.ImageNet(
        dataPath, split=dataType, download=False, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        imagenet_data, batch_size=batch_size, shuffle=True, num_workers=10, worker_init_fn=_init_fn)

    # def _save_data(data, label):
    #     global images_total, labels_total, data_per_files, data_count
    #     images_total = np.append(images_total, data, axis=0)
    #     labels_total = np.append(labels_total, label, axis=0)
    #     if images_total.shape[0] > data_per_files:
    #         np.save(savePath+dataType +
    #                 '_images_{}'.format(data_count), images_total)
    #         np.save(savePath+dataType +
    #                 '_labels_{}'.format(data_count), labels_total)
    #         images_total = np.empty((0, 224, 224, 3))
    #         labels_total = np.array([])

    def check_finished():
        for num in class_count:
            if num < data_per_class:
                return False
        return True

    for n_batch, (data, label) in enumerate(data_loader):
        data, label = data.numpy(), label.numpy()
        del_components = []

        if check_finished():
            return

        for i in range(batch_size):
            if class_count[label[i]] < data_per_class:
                class_count[label[i]] += 1
            else:
                del_components.append(i)
        if len(del_components) == batch_size:
            continue

        data = np.delete(data, del_components, axis=0)
        label = np.delete(label, del_components, axis=0)

        # if save == True:
        #     _save_data(data, label)

        # print("batch {}/{} is loaded!".format(n_batch+1, num_batch))
        yield data.transpose(0, 2, 3, 1), label


print("batch saving start...")
for n_batch, (batch_image, batch_label) in enumerate(_load_imagenet()):
    _save_data(batch_image, batch_label)
    print("batch {}/{} is loaded!".format(n_batch+1, num_batch))
_save_data([], [], finished=True)
print("batch saving Finished!")

# print("TSNE: transform start...")
# tsne = TSNE(n_components=2)
# images_tsne = tsne.fit_transform(images_total)
# print("TSNE: transform finished!")
#
# # 저장
# np.save(savePath+dataType+'_tsne_images', images_tsne)
# print("TSNE: file saved")
