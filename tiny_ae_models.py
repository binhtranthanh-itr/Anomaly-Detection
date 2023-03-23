import numpy as np
import tensorflow as tf
from tensorflow import keras


class AutoencoderModels:
    def auto_conv_tiny(self,
                       signal_length,
                       signal_width,
                       signal_height,
                       filters_net=None,
                       kernel_size=None,
                       latent_dim=4,
                       decode_activation='none',
                       name_input='auto_conv_tiny'):

        if filters_net is None:
            filters_net = [8, 8]

        input_sig = keras.Input(shape=(signal_length,), name='encoder_input')
        input_img = keras.layers.Reshape((signal_width, signal_height, 1), name="encoder_input_reshape")(input_sig)
        x = keras.layers.Conv2D(filters_net[0],
                                (kernel_size[0], kernel_size[1]),
                                strides=(2, 2),
                                padding='same',
                                activation='relu',
                                name='encoder_input_stage')(input_img)

        for i, f in enumerate(filters_net[1:]):
            x = keras.layers.Conv2D(f,
                                    (kernel_size[0], kernel_size[1]),
                                    strides=(2, 2),
                                    padding='same',
                                    activation='relu',
                                    name='encoder_stage_{}'.format(i + 1))(x)

        x = keras.layers.Flatten(name='encoder_flatten')(x)
        z = keras.layers.Dense(latent_dim, name='encoder_dense')(x)
        encoder = keras.Model(inputs=input_sig, outputs=z, name='tiny_encoder')
        encoder.summary()

        latent_inputs = keras.Input((latent_dim,), name='decoder_input')

        y = keras.layers.Dense((int(signal_width / pow(2, filters_net.size)) *
                                int(signal_height / pow(2, filters_net.size)) *
                                filters_net[-1]),
                               activation='relu',
                               name='decoder_dense')(latent_inputs)

        y = keras.layers.Reshape(target_shape=(int(signal_width / pow(2, filters_net.size)),
                                               int(signal_height / pow(2, filters_net.size)),
                                               filters_net[-1]), name='decoder_input_reshape')(y)

        for i, f in reversed(list(enumerate(filters_net[:-1]))):
            y = keras.layers.Conv2DTranspose(f,
                                             (kernel_size[0], kernel_size[1]),
                                             strides=(2, 2),
                                             padding='same',
                                             activation='relu',
                                             name='decoder_stage_{}'.format(i + 1))(y)

        if decode_activation is None or len(decode_activation) == 0 or 'none' == decode_activation:
            y = keras.layers.Conv2DTranspose(1,
                                             (kernel_size[0], kernel_size[1]),
                                             strides=(2, 2),
                                             padding='same',
                                             name='decoder_output_stage')(y)
        else:
            y = keras.layers.Conv2DTranspose(1,
                                             (kernel_size[0], kernel_size[1]),
                                             strides=(2, 2),
                                             padding='same',
                                             activation='{}'.format(decode_activation),
                                             name='decoder_output_stage')(y)

        y = keras.layers.Reshape((signal_length,), name='decoder_output_reshape')(y)

        decoder = keras.Model(inputs=latent_inputs, outputs=y, name='tiny_decoder')
        decoder.summary()

        # Define AE model.
        embedding = encoder(input_sig)
        reconstruction = decoder(embedding)
        autoencoder = keras.Model(inputs=input_sig, outputs=reconstruction, name=name_input)
        autoencoder.summary()
        return autoencoder

# def create_ae_model(signal_length, signal_width, signal_height, latent_dim):
#     # Define encoder model.
#     input_sig = keras.Input(shape=(signal_length,), name='encoder_input')
#     input_img = keras.layers.Reshape((signal_width, signal_height, 1), name="input_reshape")(input_sig)
#     x = keras.layers.Conv2D(8, (3, 3), strides=(2, 2), padding='same', activation='relu')(input_img)
#     x = keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
#     x = keras.layers.Flatten()(x)
#     x_mean = keras.layers.Dense(latent_dim, name='dense_mean')(x)
#     encoder = keras.Model(inputs=input_sig, outputs=x_mean, name='encoder')
#     encoder.summary()
#
#     # Define decoder model.
#     latent_inputs = keras.Input((latent_dim,))
#     x = keras.layers.Dense(12 * 10 * 16, activation='relu')(latent_inputs)
#     x = keras.layers.Reshape(target_shape=(12, 10, 16))(x)
#     x = keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
#     x = keras.layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
#     img_outputs = keras.layers.Conv2D(1, (3, 3), padding='same')(x)
#     decoder_outputs = keras.layers.Reshape((signal_length,))(img_outputs)
#     decoder = keras.Model(inputs=latent_inputs, outputs=decoder_outputs, name='decoder')
#     decoder.summary()
#
#     # Define AE model.
#     _x_mean = encoder(input_sig)
#     sig_outputs = decoder(_x_mean)
#     vae = keras.Model(inputs=input_sig, outputs=sig_outputs, name='ae')
#     vae.summary()
#     return vae
#
#
# def create_ae2_model(signal_length, signal_width, signal_height, latent_dim):
#     # Define encoder model.
#     input_sig = keras.Input(shape=(signal_length,), name='encoder_input')
#     input_img = keras.layers.Reshape((signal_width, signal_height, 1), name="input_reshape")(input_sig)
#     x = keras.layers.Conv2D(8, (3, 3), strides=(2, 2), padding='same', activation='relu')(input_img)
#     x = keras.layers.Conv2D(12, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
#     x = keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
#     x = keras.layers.Flatten()(x)
#     latent_inputs = keras.layers.Dense(latent_dim, name='encode_output')(x)
#
#     y = keras.layers.Dense(6 * 5 * 16, activation='relu', name='decode_input')(latent_inputs)
#     y = keras.layers.Reshape(target_shape=(6, 5, 16))(y)
#     y = keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(y)
#     y = keras.layers.Conv2DTranspose(12, (3, 3), strides=(2, 2), padding='same', activation='relu')(y)
#     y = keras.layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', activation='relu')(y)
#     y = keras.layers.Conv2DTranspose(1, (3, 3), padding='same')(y)
#     decoder_outputs = keras.layers.Reshape((signal_length,))(y)
#     ae = keras.Model(inputs=input_sig, outputs=decoder_outputs, name='autoencoder')
#     ae.summary()
#     return ae
#
#
# def create_ae3_model(signal_length, signal_width, signal_height, latent_dim):
#     # Define encoder model.
#     input_sig = keras.Input(shape=(signal_length,), name='encoder_input')
#     input_img = keras.layers.Reshape((signal_width, signal_height, 1), name="input_reshape")(input_sig)
#     x = keras.layers.Conv2D(8, (3, 3), strides=(2, 2), padding='same', activation='relu')(input_img)
#     x = keras.layers.Conv2D(12, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
#     x = keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
#     x = keras.layers.Flatten()(x)
#     latent_inputs = keras.layers.Dense(latent_dim, name='encode_output')(x)
#
#     y = keras.layers.Dense(6 * 5 * 16, activation='relu', name='decode_input')(latent_inputs)
#     y = keras.layers.Reshape(target_shape=(6, 5, 16))(y)
#     y = keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(y)
#     y = keras.layers.Conv2DTranspose(12, (3, 3), strides=(2, 2), padding='same', activation='relu')(y)
#     y = keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='relu')(y)
#     # y = keras.layers.Conv2DTranspose(1, (3, 3), padding='same')(y)
#     decoder_outputs = keras.layers.Reshape((signal_length,))(y)
#     ae = keras.Model(inputs=input_sig, outputs=decoder_outputs, name='autoencoder')
#     ae.summary()
#     return ae
