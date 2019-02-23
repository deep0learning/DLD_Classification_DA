import os
import sys
import numpy as np
import pickle


def loadPickle(pklFilePath, pklFileName):
    with open(pklFilePath + pklFileName, 'rb') as f:
        message = pickle.load(f)

    return message


def savePickle(dataArray, filePath, fileName):
    if not os.path.isdir(filePath):
        os.makedirs(filePath)

    with open(filePath + fileName, 'wb') as f:
        pickle.dump(dataArray, f)


def sortVariousPairs(pairList):
    return sorted(pairList, key=lambda x: x[1])


def getFileNameList(filePath):
    l = os.listdir(filePath)
    l = sorted(l, key=lambda x: x[:x.find('.')])

    return l


def onehotEncoder(lib_array, num_class):
    num = lib_array.shape[0]
    onehot_array = np.zeros((num, num_class))

    for i in range(num):
        onehot_array[i][lib_array[i]] = 1

    return onehot_array


def random_crop(image_batch, PADDING_SIZE=4, PAD_VALUE=-1):
    new_batch = []
    pad_width = ((PADDING_SIZE, PADDING_SIZE), (PADDING_SIZE, PADDING_SIZE), (0, 0))

    for i in range(image_batch.shape[0]):
        new_batch.append(image_batch[i])
        new_batch[i] = np.pad(image_batch[i], pad_width=pad_width, mode='constant', constant_values=PAD_VALUE)
        x_offset = np.random.randint(low=0, high=2 * PADDING_SIZE + 1, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * PADDING_SIZE + 1, size=1)[0]
        new_batch[i] = new_batch[i][x_offset:x_offset + 32, y_offset:y_offset + 32, :]

    return new_batch


def random_flip(image_batch):
    for i in range(image_batch.shape[0]):
        flip_prop = np.random.randint(low=0, high=3)
        if flip_prop == 0:
            image_batch[i] = image_batch[i]
        if flip_prop == 1:
            image_batch[i] = np.fliplr(image_batch[i])
        if flip_prop == 2:
            image_batch[i] = np.flipud(image_batch[i])

    return image_batch


def next_batch(image, label, batch_size):
    index = np.random.randint(low=0, high=len(image), size=batch_size)
    img_batch = image[index]
    lab_batch = label[index]
    img_batch = random_flip(img_batch)
    img_batch = random_crop(img_batch)

    return img_batch, lab_batch


def next_batch_unpaired(image, batch_size):
    index = np.random.randint(low=0, high=len(image), size=batch_size)
    img_batch = image[index]
    img_batch = random_flip(img_batch)
    img_batch = random_crop(img_batch)

    return img_batch
