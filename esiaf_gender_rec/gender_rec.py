from speechemotionrecognition.dnn import CNN, LSTM
from speechemotionrecognition.mlmodel import SVM, RF, NN
import numpy as np
from speechpy.feature import mfcc
import os

net_dict = {'CNN': CNN, 'LSTM': LSTM, 'SVM': SVM, 'RF': RF, 'NN': NN}
labels = ('Male', 'Female')

class Gender_rec:

    def __init__(self, args):
        if args['model_type'] not in net_dict:
            print('Model type not supported!')
            exit(1)
        absolute_model_path = os.path.expandvars(args['model_save_file'])
        if not os.path.isfile(absolute_model_path):
            print('Model save file was not found!')
            exit(1)
        shape = (35, 39)
        self.model = net_dict[args['model_type']](input_shape=shape, num_classes=len(labels))
        self.model.restore_model(absolute_model_path)
        self.model.model._make_predict_function() # needed to fix a bug caused apparently by creating two models
        self.mean_signal_length = 6000

    def get_mfccs(self, signal):
        s_len = len(signal)

        if s_len < self.mean_signal_length:
            pad_len = self.mean_signal_length - s_len
            pad_rem = pad_len % 2
            pad_len //= 2
            signal = np.pad(signal, (pad_len, pad_len + pad_rem),
                            'constant', constant_values=0)
        else:
            pad_len = s_len - self.mean_signal_length
            pad_len //= 2
            signal = signal[pad_len:pad_len + self.mean_signal_length]
        mel_coefficients = mfcc(signal, 16000, num_cepstral=39)
        return mel_coefficients

    def recognize_gender(self, audio):
        # prepare data
        mfccs = self.get_mfccs(audio)
        # call model and return label
        prediction = self.model.predict_one(mfccs)
        best = np.argmax(prediction)
        return labels[best], prediction[best]