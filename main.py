import python_speech_features
from data_generator import vis_train_features, plot_spectrogram_feature
from IPython.display import Markdown, display, Audio
from data_generator import vis_train_features, plot_raw_audio

# import NN architectures for speech recognition
from models import *
# import function for training acoustic model
from train_utils import train_model

import numpy as np
from data_generator import AudioGenerator
from keras import backend as K
from utils import int_sequence_to_text

from keras.optimizers.legacy import SGD

# extract label and audio features for a single training example
vis_text, vis_raw_audio, vis_mfcc_feature, vis_spectrogram_feature, vis_audio_path = vis_train_features()

# plot normalized spectrogram
plot_spectrogram_feature(vis_spectrogram_feature)
# print shape of spectrogram
display(Markdown('**Shape of Spectrogram** : ' + str(vis_spectrogram_feature.shape)))


model_0 = simple_rnn_model(input_dim=161)
train_model(input_to_softmax=model_0,
            pickle_path='model_0.pickle',
            save_model_path='model_0.h5',
            spectrogram=True) 

model_1 = rnn_model(input_dim=161, 
                    units=200,
                    activation='relu')

train_model(input_to_softmax=model_1,
            pickle_path='model_1.pickle',
            save_model_path='model_1.h5',
            spectrogram=True) 

model_2 = cnn_rnn_model(input_dim=161, 
                        filters=200,
                        kernel_size=11,
                        conv_stride=2,
                        conv_border_mode='valid',
                        units=200)

train_model(input_to_softmax=model_2,
            pickle_path='model_2.pickle',
            save_model_path='model_2.h5',
            spectrogram=True,optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1))

model_3 = deep_rnn_model(input_dim=161, 
                         units=200,
                         recur_layers=2)

train_model(input_to_softmax=model_3,
            pickle_path='model_3.pickle',
            save_model_path='model_3.h5',
            spectrogram=True,
            optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1))

model_4 = bidirectional_rnn_model(input_dim=161, 
                                  units=200)


train_model(input_to_softmax=model_4,
            pickle_path='model_4.pickle',
            save_model_path='model_4.h5',
            spectrogram=True,
           optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1)) 



def get_predictions(index, partition, input_to_softmax, model_path):
    """ Print a model's decoded predictions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        input_to_softmax (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """
    # load the train and test data
    data_gen = AudioGenerator()
    data_gen.load_train_data()
    data_gen.load_validation_data()

    # obtain the true transcription and the audio features
    if partition == 'validation':
        transcr = data_gen.valid_texts[index]
        audio_path = data_gen.valid_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    elif partition == 'train':
        transcr = data_gen.train_texts[index]
        audio_path = data_gen.train_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    else:
        raise Exception('Invalid partition!  Must be "train" or "validation"')

    # obtain and decode the acoustic model's predictions
    input_to_softmax.load_weights(model_path)
    prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    output_length = [input_to_softmax.output_length(data_point.shape[0])]
    pred_ints = (K.eval(K.ctc_decode(
        prediction, output_length)[0][0]) + 1).flatten().tolist()

    # play the audio file, and display the true and predicted transcriptions
    print('-' * 80)
    Audio(audio_path)
    print('True transcription:\n' + '\n' + transcr)
    print('-' * 80)
    print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints)))
    print('-' * 80)


get_predictions(index=0,
                partition='train',
                input_to_softmax=final_model(input_dim=161, 
                                  units=200, dropout=0.2),
                model_path='results/model_end.h5')