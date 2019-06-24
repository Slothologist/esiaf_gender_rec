import soundfile
from os import listdir
from os.path import isfile, join
from speechpy.feature import mfcc
import numpy as np
import random

from keras.utils import np_utils

from speechemotionrecognition.dnn import LSTM


def _read_file(file, framesize):
    blocks = []
    for block in soundfile.blocks(file, blocksize=framesize):
        if len(block) == framesize:
            blocks.append(block)
    return blocks


def _calc_mfcc(wav):
    mfccs = mfcc(wav, 16000, num_cepstral=39)
    return mfccs


def _read_dataset(path):
    male = join(path, 'male')
    female = join(path, 'female')
    male_files = [join(male, f) for f in listdir(male) if isfile(join(male, f))]
    female_files = [join(female, f) for f in listdir(female) if isfile(join(female, f))]
    return male_files, female_files


def _shuffle(x, y):
    zipped_list = zip(x, y)
    random.shuffle(zipped_list)
    return zip(*zipped_list)



def prepare_training_data(data_path, block_size):
    male_files, female_files = _read_dataset(data_path)
    male_wavs = [singular_wav for file in male_files for singular_wav in _read_file(file, block_size)]  # puts all wavs from males in a 1-dim list
    female_wavs = [singular_wav for file in female_files for singular_wav in _read_file(file, block_size)]
    male_mfccs = [_calc_mfcc(wav) for wav in male_wavs]
    female_mfccs = [_calc_mfcc(wav) for wav in female_wavs]
    if not len(male_mfccs) == len(female_mfccs):
        male_mfccs , _ = _shuffle(male_mfccs, [1 for x in male_mfccs])
        female_mfccs, _ = _shuffle(female_mfccs, [1 for x in female_mfccs])
        common = min(len(male_mfccs), len(female_mfccs))
        male_mfccs = male_mfccs[:common]
        female_mfccs = female_mfccs[:common]
    data = male_mfccs + female_mfccs
    labels = [0 for _ in male_mfccs] + [1 for _ in female_mfccs]
    return np.array(data), np.array(labels)



if __name__ == '__main__':
    train_test_factor = 0.8

    # get training data
    data, labels = prepare_training_data('/home/rfeldhans/programming/audio/esiaf_gender_rec/dataset/', 6000)


    # schuffle for train/test assignment
    x, y = _shuffle(data, labels)

    x_train = np.array(x[:int(len(x)*train_test_factor)])
    y_train = np.array(y[:int(len(y)*train_test_factor)])
    x_test = np.array(x[int(len(x)*train_test_factor):])
    y_test = np.array(y[int(len(y)*train_test_factor):])

    # prepare labels
    y_train = np_utils.to_categorical(y_train)
    y_train_test = np_utils.to_categorical(y_test)

    model = LSTM(input_shape=x_train[0].shape,
                 num_classes=2)

    model.train(x_train, y_train, x_test, y_train_test, n_epochs=20)
    model.evaluate(x_test, y_test)
    model.save_model()

