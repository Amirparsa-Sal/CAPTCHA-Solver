from typing import List, Tuple
import numpy as np
import tensorflow as tf
import tensorflow.io as tfio
import tensorflow.image as tfim
import tensorflow.data as tfd
import math

def find_number_of_characters(captchas: List[str]):
    '''
    Finds the number of different characters used in the dataset.
    :param captchas: A list of strings, each string is a file name.
    :return: A set of characters.
    '''
    chars = set()
    for name in captchas:
        captcha = name.split('.')[0]
        for char in captcha:
            chars.add(char)
    return chars

def preprocess(image_path: tf.Tensor, image_size = (0, 0)) -> tf.Tensor:
  '''
  A function to preprocess an image.
  :param image_path tf.Tensor: path of the image stored in a tensor
  :param image_size int: output image size if reshaping is needed. by default it will not reshape the image.
  :return The tensor of the normalized image.
  '''
  # read image file
  image = tfio.read_file(image_path)
  # decode the image into 3 channels
  image = tfim.decode_jpeg(image, channels = 3)
  # black and white
  image = tfim.rgb_to_grayscale(image)
  # normalize the image
  image = tfim.convert_image_dtype(image, np.float32)
  # reshape the image if needed
  if image_size is not None:
    image = tfim.resize(image, size = image_size)
  return image
  
def create_label_matrix(captcha_value: str, chars: np.array) -> List[np.array]:
  '''
  Creates a numpy array with size 1 * len(captcha_velue) * len(chars) containing one-hot encoded values
  of the captcha_value characters with respecto chars.
  :param captcha_value str: a string containg correct answer for a capctha
  :param chars np.array: a numpy array containing all characters used in captchas
  :return a numpy array with size 1 * len(captcha_velue) * len(chars) containing one-hot encoded values
  '''
  length = len(captcha_value)
  digits = []
  for i in range(length):
    digits.append((chars == captcha_value[i]).astype(int))
  return digits

def create_datasets(file_names: List[str], labels: List[str], chars: np.array, shuffle: bool = True,
                    test_size: float = 0.1, val_size: float = 0.1, image_size: Tuple[int,int] = None):
    '''
    A function to create train, validation and test datasets.
    :param file_names List[str]: a list of strings containing file names
    :param labels List[str]: a list of strings containing labels
    :param chars np.array: a numpy array containing all characters used in captchas
    :param shuffle bool: if the dataset should be shuffled or not
    :param test_size float: the size of the test dataset
    :param val_size float: the size of the validation dataset
    :param image_size Tuple[int,int]: the size of the images
    :return train, validation and test datasets
    '''
    # creating the labels
    new_labels = [[] for i in range(5)]
    for l in labels:
        labels = create_label_matrix(l, chars)
        for i in range(5):
            new_labels[i].append(labels[i])
    # create the dataset
    data = (file_names, ) + tuple(new_labels)
    dataset = tfd.Dataset.from_tensor_slices(data)
    # shuffle if needed
    if shuffle:
        dataset = dataset.shuffle(buffer_size = len(file_names), reshuffle_each_iteration = False)
    # load images and preprocess
    dataset = dataset.map(lambda filename, l0, l1, l2, l3, l4: (preprocess(filename, image_size = image_size), l0, l1, l2, l3, l4))
    # create training, validation and test dataset
    train_size = math.floor(len(file_names) * (1 - val_size - test_size))
    val_size = math.floor(len(file_names) * val_size)
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.take(train_size + val_size).skip(train_size)
    test_dataset = dataset.skip(train_size + val_size)
    return train_dataset, val_dataset, test_dataset

def generate_batch(dataset: tfd.Dataset, batch_size : int = 32, test = False, shuffle = True, shuffle_buffer_size = 10):
    '''
    A function to generate batches from a dataset.
    :param dataset tfd.Dataset: a dataset to generate batches from
    :param batch_size int: the size of the batch
    :param shuffle bool: if the dataset should be shuffled or not
    :param shuffle_buffer_size int: the size of the buffer used to shuffle the dataset
    :return a batch from the dataset
    '''
    images, l0s, l1s, l2s, l3s, l4s = [], [], [], [], [], []
    while True:
        # shuffle the dataset if needed
        if shuffle:
            dataset = dataset.shuffle(buffer_size = shuffle_buffer_size)
            # create batches
            for img, l0, l1, l2, l3, l4 in dataset:
                images.append(img)
                l0s.append(l0)
                l1s.append(l1)
                l2s.append(l2)
                l3s.append(l3)
                l4s.append(l4)
                # if batch is full, yield it
                if len(images) == batch_size:
                    yield np.array(images), [np.array(l0s), np.array(l1s), np.array(l2s), np.array(l3s), np.array(l4s)]
                    images, l0s, l1s, l2s, l3s, l4s = [], [], [], [], [], []
