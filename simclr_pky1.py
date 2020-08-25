
import matplotlib.pyplot as plt
import matplotlib
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import re
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def count_params(checkpoint, excluding_vars=[], verbose=True):
    vdict = checkpoint.get_variable_to_shape_map()
    cnt = 0
    for name, shape in vdict.items():
        skip = False
        for evar in excluding_vars:
            if re.search(evar, name):
                skip = True
        if skip:
            continue
        if verbose:
            print(name, shape)
        cnt += np.prod(shape)
    cnt = cnt / 1e6
    print("Total number of parameters: {:.2f}M".format(cnt))
    return cnt


checkpoint_path = 'gs://simclr-checkpoints/simclrv2/finetuned_100pct/r50_1x_sk0/'
checkpoint = tf.train.load_checkpoint(checkpoint_path)
_ = count_params(checkpoint, excluding_vars=[
                 'global_step', "Momentum", 'ema', 'memory', 'head'], verbose=False)


imagenet_int_to_str = {}

with open('ilsvrc2012_wordnet_lemmas.txt', 'r') as f:
    for i in range(1000):
        row = f.readline()
        row = row.rstrip()
        imagenet_int_to_str.update({i: row})

tf_flowers_labels = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']


def preprocess_for_train(image, height, width, color_distort=True, crop=True, flip=True):
    """Preprocesses the given image for training.
    Args:
      image: `Tensor` representing an image of arbitrary size.
      height: Height of output image.
      width: Width of output image.
      color_distort: Whether to apply the color distortion.
      crop: Whether to crop the image.
      flip: Whether or not to flip left and right of an image.
    Returns:
      A preprocessed image `Tensor`.
    """
    if crop:
        image = random_crop_with_resize(image, height, width)
    if flip:
        image = tf.image.random_flip_left_right(image)
    if color_distort:
        image = random_color_jitter(image)
    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)
    return image


def preprocess_for_eval(image, height, width, crop=True):
    """Preprocesses the given image for evaluation.
    Args:
      image: `Tensor` representing an image of arbitrary size.
      height: Height of output image.
      width: Width of output image.
      crop: Whether or not to (center) crop the test images.
    Returns:
      A preprocessed image `Tensor`.
    """
    if crop:
        image = center_crop(image, height, width,
                            crop_proportion=CROP_PROPORTION)
    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)
    return image


def preprocess_image(image, height, width, is_training=False, color_distort=True, test_crop=True):
    """Preprocesses the given image.
    Args:
      image: `Tensor` representing an image of arbitrary size.
      height: Height of output image.
      width: Width of output image.
      is_training: `bool` for whether the preprocessing is for training.
      color_distort: whether to apply the color distortion.
      test_crop: whether or not to extract a central crop of the images
          (as for standard ImageNet evaluation) during the evaluation.
    Returns:
      A preprocessed image `Tensor` of range [0, 1].
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if is_training:
        return preprocess_for_train(image, height, width, color_distort)
    else:
        return preprocess_for_eval(image, height, width, test_crop)


batch_size = 5
dataset_name = 'tf_flowers'
tfds_dataset, tfds_info = tfds.load(
    dataset_name, split='train', with_info=True)
num_images = tfds_info.splits['train'].num_examples
num_classes = tfds_info.features['label'].num_classes


def _preprocess(x):
    x['image'] = preprocess_image(
        x['image'], 224, 224, is_training=False, color_distort=False)
    return x


x = tfds_dataset.map(_preprocess).batch(batch_size)
x = tf.data.make_one_shot_iterator(x).get_next()


hub_path = 'gs://simclr-checkpoints/simclrv2/finetuned_100pct/r50_1x_sk0/hub/'
module = hub.Module(hub_path, trainable=False)
key = module(inputs=x['image'], signature="default", as_dict=True)
logits_t = key['logits_sup'][:, :]


sess = tf.Session()
sess.run(tf.global_variables_initializer())


image, labels, logits = sess.run((x['image'], x['label'], logits_t))
pred = logits.argmax(-1)


fig, axes = plt.subplots(5, 1, figsize=(15, 15))
for i in range(5):
    axes[i].imshow(image[i])
    true_text = tf_flowers_labels[labels[i]]
    pred_text = imagenet_int_to_str[pred[i]]
    if i == 0:
        axes[i].text(
            0, 0, 'Attention: the predictions here are inaccurate as they are constrained among 1000 ImageNet classes.\n', c='r')
    axes[i].axis('off')
    axes[i].text(256, 128, 'Truth: ' + true_text + '\n' + 'Pred: ' + pred_text)
