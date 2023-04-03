import sys

import tensorflow as tf
from tensorflow.keras import backend as K
from tqdm import trange
import matplotlib.pyplot as plt

from layers import *


def get_output_from_pretrained_model(pretrain_checkpoint_dir, input_data):
    with tf.device('/cpu:0'):
        test_model = tf.keras.models.load_model(pretrain_checkpoint_dir)
        encoder = test_model.layers[1]
        decoder = test_model.layers[2]
        specific_layer = decoder.layers[-3]
        func = K.function([test_model.get_layer(index=0).input], encoder.output)
        latents = func([input_data])
        dfunc = K.function([decoder.input], specific_layer.output)
        layer_output = dfunc([latents])

    return layer_output


if __name__ == '__main__':
    ####################### CONFIG #######################
    pretrain_checkpoint_dir = '/home/ai_dev_02/Documents/quoctn_itrvn/ITR-projects/ITR-sound/dataset/20230321093220' \
                              '/output/models/auto_conv_tiny_8_8.8.8_3.3_relu_128_0.001_1679466516/saved_autoencoder_model'
    save_numpy_model = 'custom_conv_decoder_k'
    learning_rate = 1e-3
    batch_size = 1
    epoch_num = 1

    ####################### PREPARE DATA #######################
    raw_data = np.load('/home/ai_dev_02/Documents/quoctn_itrvn/ITR-projects/ITR-TinyML/test_autoencoder_scratch/custom_data/abnormal_spectral.npy')
    raw_data = np.reshape(raw_data, (raw_data.shape[0], 1920))
    black_box_data_as_input = get_output_from_pretrained_model(pretrain_checkpoint_dir, raw_data)
    train_data = np.transpose(black_box_data_as_input, (0, 3, 1, 2))  # B, C, H, W

    ####################### DEFINE AND TRAIN MODEL #######################
    model = Network()
    model.add_layer(ConvLayer(n_filters=4, filter_shape=(3, 3), stride=(1, 1), padding=1)).add_layer(Activation('relu')) \
        .add_layer(ReshapeLayer((1920,))) \
        .add_layer(MAELayer())

    for epoch in range(epoch_num):
        train_order = np.random.permutation(len(train_data))
        bar = trange(len(train_data), file=sys.stdout)
        running_loss = 0
        for i in bar:
            loss = model.forward(train_data[train_order[i]][np.newaxis, ...], raw_data[train_order[i]][np.newaxis, ...])
            bar.set_description('Loss: {:.4f}'.format(loss))
            model.backward()
            model.adam_trainstep(alpha=learning_rate)
            running_loss += loss
        print('Final loss {:.4f}'.format(running_loss / len(train_data)))

        # save model
        model.save(save_numpy_model)

    ####################### TEST MODEL #######################

    model = Network()
    model.load(save_numpy_model)
    for i in range(1):
        temp = np.expand_dims(train_data[i], axis=0)
        recons = model.run(temp, k=-1)[0]

        # plot in signal format
        plt.figure()
        plt.plot(raw_data[i], label='origin')
        plt.plot(recons, label='reconstruct')
        plt.legend()

        # plot in spectrogram format
        plt.figure()
        plt.subplot(1, 2, 1)
        img = np.reshape(raw_data[i], (48, 40))
        plt.imshow(img)
        plt.title('origin')
        plt.subplot(1, 2, 2)
        img_recon = np.reshape(recons, (48, 40))
        plt.imshow(img_recon)
        plt.title('reconstruct')

        plt.show()
