import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import json

from glob import glob
from random import shuffle
from functools import partial
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Dropout, Flatten, MaxPool1D, Bidirectional, Concatenate, \
    Dense, concatenate, Conv2DTranspose, UpSampling2D, Conv1DTranspose, BatchNormalization, Input, Conv2D, MaxPool2D, \
    MaxPooling2D, LSTM, Add, Reshape
from tiny_ae_models import AutoencoderModels
from loss.losses import CategoricalCrossEntropy, Root_Mean_Squared_Error, Mean_Squared_Error, Mean_Absolute_Error


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def _preprocess_proto(example_proto, feature_len, class_num):
    """Read sample from protocol buffer."""
    encoding_scheme = {
        'sample': tf.io.FixedLenFeature(shape=[feature_len, ], dtype=tf.int64),
        'label': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
    }
    proto = tf.io.parse_single_example(example_proto, encoding_scheme)
    sample = proto['sample']
    label = proto["label"]
    return sample, sample

def _get_tfrecord_filenames(dir_path, is_training):
    if not os.path.exists(dir_path):
        raise FileNotFoundError("{}; No such file or directory.".format(dir_path))

    filenames = sorted(glob(os.path.join(dir_path, "*.tfrecord")))
    if not filenames:
        raise FileNotFoundError("No TFRecords found in {}".format(dir_path))

    if is_training:
        shuffle(filenames)

    return filenames

def get_dataset(dir_path,
                feature_len,
                class_num):
    """Load TFRecords for training and evaluation on PC."""

    filenames = _get_tfrecord_filenames(dir_path, False)
    dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.map(partial(_preprocess_proto,
                                  feature_len=feature_len,
                                  class_num=class_num),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(1)
    return dataset

def load_architecture_model():
    decode_activation = 'relu'
    model_name = 'auto_conv_tiny'

    feature_height = 40
    feature_width = 48
    feature_len = 1920
    num_filters = np.array([8, 8, 8])
    kernel_size = np.array([3, 3])
    latent_dim = 8
    sound_models = AutoencoderModels()
    train_model = getattr(sound_models, model_name)(feature_len,
                                                    feature_width,
                                                    feature_height,
                                                    num_filters,
                                                    kernel_size,
                                                    latent_dim,
                                                    decode_activation)


def build_new_model(saved_encoder_model_dir = '20230321093220/output/models/auto_conv_tiny_8_8.8.8_3.3_relu_128_0.001_1679466516/saved_autoencoder_model'):
    # Load tiny_ae_model
    # saved_encoder_model_dir = '/mnt/Data_1/Project/sound_detection/20230321093220/output/models/auto_conv_tiny_8_8.8.8_3.3_relu_128_0.001_1679466516/saved_autoencoder_model'

    model = tf.keras.models.load_model(saved_encoder_model_dir)
    # Encode part
    x = model.layers[1].output
    # Decode - remove last CNN tranpose layer
    for i in range(1, 5, 1):
        x = model.layers[2].layers[i](x)

    model_output = x
    new_model = Model(model.input, model_output, name='Tiny-ae')
    new_model.summary()

    return new_model


def fine_tune_model():
    model_input = Input(shape=(24, 20, 8))
    x = model_input
    x = Conv2D(4, (1, 1), (1, 1), padding='same', activation='relu')(x)

    # x = Conv2DTranspose(1,
    #                     (8, 8),
    #                     strides=(2, 2),
    #                     padding='same',
    #                     activation='relu',
    #                     name='decoder_output_stage')(x)

    x = Flatten()(x)
    model_output = x
    model = Model(model_input, model_output)
    model.summary()
    return model

##############################################################
from layers.convolution import Convolution as Self_Convolution
from layers.flatten import Flatten as Self_Flatten
from layers.activation import Relu as Self_Relu
from utilities.self_model import Self_Model

def fine_tune_model_2():
    model = Self_Model(
        Self_Convolution(kernel_shape=(3, 3), filters=4, padding='same'),
        Self_Relu(),
        Self_Flatten(),
        loss=Mean_Absolute_Error,
        batch_size=1,
        num_classes=1920,
        name='cnn'
    )

    return model

##############################################################

if __name__ == '__main__':
    saved_encoder_model_dir = '20230321093220/output/models/auto_conv_tiny_8_8.8.8_3.3_relu_128_0.001_1679466516/saved_autoencoder_model'
    abnormal_directory = '/mnt/Project/ECG/Source_Dong/project_test/Anomaly-Detection/20230321093220/data/eval-abnormal/'
    eval_directory = '/mnt/Project/ECG/Source_Dong/project_test/Anomaly-Detection/20230321093220/data/eval/'

    epochs = 100
    len_statistic = 100

    feature_len = 1920
    type_num = 3
    dataset = get_dataset(dir_path=abnormal_directory,
                          feature_len=feature_len,
                          class_num=type_num)
    inputs = list(dataset.as_numpy_iterator())
    inputs = [ar[0].flatten() for ar in inputs]
    inputs = np.array(inputs)
    inputs = np.array(inputs)[2, :]
    inputs = inputs.reshape(1, 1920)

    # load tiny_ae_model without last layer
    ae_model = build_new_model(saved_encoder_model_dir)
    input_fine_tune = ae_model.predict(inputs)

    # fine_tune_model (last layer of decode)
    ft_model = fine_tune_model_2()

    mae_statistic = np.zeros(len_statistic, dtype=int)
    cnt_init = 0
    cnt_reset = 0

    print('Training Processing')
    i=0
    while i < epochs:
        # print("i: ", i)
        ft_model.train(input_fine_tune, inputs.T, batch_size=1, epochs=1, learning_rate=0.1)

        # input_fine_tune = ae_model.predict(inputs)
        evaluate, prediction = ft_model.evaluate(input_fine_tune, inputs.T)

        if evaluate > 100 and i < 2:
            if cnt_reset == 0:
                print(f"INIT MODEL {cnt_reset} - RESET INIT: {cnt_init}")
            else:
                print(f"RESET MODEL {cnt_reset} - RESET INIT: {cnt_init}")

            cnt_init += 1
            # fine_tune_model (last layer of decode)
            ft_model = fine_tune_model_2()
            i = 0
        else:
            # mae_statistic[f'{int(i)}'] += 1
            if int(evaluate) >= len_statistic:
                mae_statistic[-1] += 1
            else:
                mae_statistic[int(evaluate)] += 1

            # print('Loss = ', evaluate)
            i += 1

        if i >= epochs and evaluate > 35:
            print("RESET MODEL: ", cnt_reset)
            cnt_reset += 1
            cnt_init = 0
            i = 0

    eval_data = get_dataset(dir_path=eval_directory,
                          feature_len=feature_len,
                          class_num=type_num)

    inputs = list(dataset.as_numpy_iterator())
    inputs = [ar[0].flatten() for ar in inputs]
    inputs = np.array(inputs)

    # if os.path.exists(eval_directory + 'log_mae.json'):
    #     file = open(eval_directory + 'log_mae.json', 'r')
    #     mae_statistic_eval = json.load(file)['eval']
    # else:
    if True:
        mae_statistic_eval = np.zeros(len_statistic, dtype=np.int)
        for i in range(inputs.shape[0]):
            input_eval = np.array(inputs)[i, :]
            input_eval = input_eval.reshape(1, 1920)

            evaluate, _ = ft_model.evaluate(input_fine_tune, input_eval.T)
            if int(evaluate) >= len_statistic:
                mae_statistic_eval[-1] += 1
            else:
                mae_statistic_eval[int(evaluate)] += 1

        file = open(eval_directory + 'log_mae.json', 'w')
        mae_statistic_eval_dict = dict()
        mae_statistic_eval_dict['eval'] = list(mae_statistic_eval)
        file.writelines(json.dumps(mae_statistic_eval_dict, default=np_encoder))

    file.close()

    # plt.figure(1)
    # plt.plot(inputs[0, :], 'r', prediction, 'b--')
    # plt.title("CNN-Python")

    # plt.figure(2)
    plt.bar(np.arange(0, len_statistic, 1), mae_statistic, label='abnormal')
    plt.bar(np.arange(0, len_statistic, 1), mae_statistic_eval, alpha=0.5, label='eval')
    plt.legend()
    plt.show()

    # ft_model.compile(optimizer='adam',
    #                  loss='mean_absolute_error')
    # ft_model.fit(x=input_fine_tune, y=inputs, epochs=1)
