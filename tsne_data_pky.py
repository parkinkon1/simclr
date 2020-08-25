
import argparse
import time
from PIL import Image
import importlib.util

import sys
import os
import yaml
import numpy as np
import subprocess
import random

from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.manifold import TSNE

# subprocess.check_call(["gcloud", "auth", "application-default", "login"])
seed = 201711075
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.


parser = argparse.ArgumentParser(description='feature extraction')
parser.add_argument('-d', '--data', required=True,
                    type=str, help='data name')
args = parser.parse_args()


filename = args.data
filePath = '/home/user/Desktop/pky/simclr/save/Imagenet/'
savePath = '/home/user/Desktop/pky/simclr/save/Imagenet/'
print('savePath:', savePath)

features = np.load(filePath+filename)

print("TSNE: transform start...")
tsne = TSNE(n_components=2, n_jobs=10)
features_tsne = tsne.fit_transform(features.reshape(features.shape[0], -1))
print("TSNE: transform finished!")

# 저장
np.save(savePath+'tsne_'+filename, features_tsne)
print("TSNE: file saved")
