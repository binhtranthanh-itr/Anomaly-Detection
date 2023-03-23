import argparse
import calendar
import csv
import datetime
import glob
import json
import logging
import math
import multiprocessing
import os
import re
import sys
import time
import warnings
import zipfile
from functools import partial
from glob import glob
from random import shuffle

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

from prettytable import PrettyTable
from matplotlib import pyplot as plt
from tiny_ae_models import AutoencoderModels

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('botocore').setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(1)


class TextLogging:
    def __init__(self, file_path, mode):
        self.file_path = file_path
        self.mode = mode

        # with open(file_path, mode) as test_file:
        #     pass

    def write_mylines(self, lines):
        with open(self.file_path, self.mode) as log_file:
            log_file.writelines(lines)


class Training:
    @staticmethod
    def _calc_num_steps(num_samples, batch_size):
        return (num_samples + batch_size - 1) // batch_size

    @staticmethod
    def _get_tfrecord_filenames(dir_path, is_training):
        if not os.path.exists(dir_path):
            raise FileNotFoundError('{}; No such file or directory.'.format(dir_path))

        filenames = sorted(glob(os.path.join(dir_path, '*.tfrecord')))
        if not filenames:
            raise FileNotFoundError('No TFRecords found in {}'.format(dir_path))

        if is_training:
            shuffle(filenames)

        return filenames

    @staticmethod
    def format_metrics(metrics, sep='; '):
        return sep.join('{}: {:.6f}'.format(k, metrics[k]) for k in sorted(metrics.keys()))

    def train_vae(self,
                  use_gpu_index,
                  model_name,
                  latent_dim,
                  num_filters,
                  kernel_size,
                  log_dir,
                  model_dir,
                  datastore_dict,
                  resume_from,
                  train_directory,
                  test_directory,
                  eval_directory,
                  batch_size,
                  learning_rate,
                  decode_activation,
                  valid_freq,
                  patience,
                  epoch_num):

        print('+ Model dir: {}'.format(model_dir))
        feature_height = datastore_dict['feature_height']
        feature_width = datastore_dict['feature_width']
        feature_len = datastore_dict['feature_len']
        saved_encoder_model_dir = model_dir + '/save_encoder_model'
        best_loss_checkpoint_dir = model_dir + '/best_autoencoder_loss'
        saved_ae_model_dir = model_dir + '/saved_autoencoder_model'

        for i in [saved_encoder_model_dir, best_loss_checkpoint_dir, saved_ae_model_dir]:
            if not os.path.exists(i):
                os.makedirs(i)

        bk_metric = None
        if os.path.exists('{}/{}_bk_metric.txt'.format(log_dir, model_name)):
            with open('{}/{}_bk_metric.txt'.format(log_dir, model_name), 'r') as json_file:
                bk_metric = json.load(json_file)
                try:
                    if bk_metric['stop_train']:
                        return True
                except (Exception,):
                    bk_metric['stop_train'] = False

        fieldnames = ['epoch',
                      'loss_train', 'reconstruction_loss_train', 'kl_loss_train',
                      'loss_eval', 'reconstruction_loss_eval', 'kl_loss_eval']

        if not os.path.exists(log_dir + '/{}_log.csv'.format(model_name)):
            with open(log_dir + '/{}_log.csv'.format(model_name), mode='a+') as report_file:
                report_writer = csv.DictWriter(report_file, fieldnames=fieldnames)
                report_writer.writeheader()

        log_train = TextLogging(log_dir + '/{}_training_log.txt'.format(model_name), 'a+')
        log_train.write_mylines('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        log_train.write_mylines('Begin : {}\n'.format(str(datetime.datetime.now())))
        log_train.write_mylines('Batch size : {}\n'.format(batch_size))
        log_train.write_mylines('Epoch : {}\n'.format(epoch_num))
        log_train.write_mylines('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(use_gpu_index)

        import tensorflow as tf
        import logging
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        tf.get_logger().setLevel(logging.ERROR)
        tf.autograph.set_verbosity(1)

        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            print('+ Physical Devices: GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        else:
            print('+ Cuda Visible Devices: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
            print('+ Physical Devices: CPU')

        class CustomCallback(tf.keras.callbacks.Callback):
            def __init__(self,
                         model_name,
                         log_train,
                         stopped_epoch,
                         saved_encoder_model_dir,
                         best_loss_checkpoint_dir,
                         saved_ae_model_dir,
                         bk_metric,
                         log_dir,
                         field_name,
                         best_loss=-1,
                         valid_freq=6,
                         patience=3,
                         length_train=None,
                         length_valid=None,
                         tensorboard_dir=None
                         ):
                super(CustomCallback, self).__init__()
                self.bk_metric = bk_metric
                self.log_train = log_train
                self.saved_encoder_model_dir = saved_encoder_model_dir
                self.best_loss_checkpoint_dir = best_loss_checkpoint_dir
                self.saved_ae_model_dir = saved_ae_model_dir
                self.model_name = model_name
                self.stopped_epoch = stopped_epoch
                self.best_loss = best_loss
                self.log_dir = log_dir
                self.fieldnames = field_name
                self.train_progressbar = None
                self.length_train = length_train
                self.length_valid = length_valid
                self.progress = 0
                self.valid_freq = valid_freq
                self.patience = patience
                self.wait = 0
                self.epoch_early_stopping = 0
                self.tensorboard_dir = tensorboard_dir
                self.best_weights = None

            def on_test_batch_end(self, batch, logs=None):
                self.progress += 1
                if self.length_train is not None and self.length_valid is not None and self.train_progressbar is not None:
                    self.train_progressbar.update(self.progress)
                    sys.stdout.flush()

            def on_train_batch_end(self, batch, logs=None):
                self.progress += 1
                if self.length_train is not None and self.train_progressbar is not None:
                    self.train_progressbar.update(self.progress)
                    sys.stdout.flush()

            def on_epoch_begin(self, epoch, logs=None):
                epoch += 1
                print("Epoch {}/{}".format(epoch, self.stopped_epoch))
                self.log_train.write_mylines("Epoch {}/{}\n".format(epoch, self.stopped_epoch))
                self.progress = 0

            def on_epoch_end(self, epoch, logs=None):
                epoch += 1
                report_row = dict()
                report_row['epoch'] = epoch
                report_row['loss_train'] = logs['loss']
                train_metrics = {
                    'loss': logs['loss'],
                }

                print('++ Training: ')
                metrics_table = PrettyTable([k for k in train_metrics.keys()])
                metrics_table.add_row([round(train_metrics[k], 6) for k in sorted(train_metrics.keys())])
                print(metrics_table)
                self.log_train.write_mylines('++ Training: \n{}\n'.format(metrics_table))
                print(metrics_table)
                self.log_train.write_mylines('{}\n'.format(metrics_table))
                if self.tensorboard_dir is not None:
                    with tf.summary.create_file_writer(self.tensorboard_dir + '/train').as_default():
                        tf.summary.scalar('loss', train_metrics['loss'], step=epoch - 1)
                # endregion Training

                if epoch >= self.valid_freq and epoch % self.valid_freq == 0:
                    # region Eval
                    print('\n\n++ Validation: ')
                    val_metrics = {
                        'loss': logs['val_loss'],
                    }

                    report_row['loss_eval'] = val_metrics['loss']
                    metrics_table = PrettyTable([k for k in val_metrics.keys()])
                    metrics_table.add_row([round(val_metrics[k], 6) for k in sorted(val_metrics.keys())])
                    print(metrics_table)
                    self.log_train.write_mylines('\n\n++ Validation: \n{}\n'.format(metrics_table))
                    self.log_train.write_mylines('{}\n'.format(metrics_table))

                    if self.tensorboard_dir is not None:
                        with tf.summary.create_file_writer(self.tensorboard_dir + '/validation').as_default():
                            tf.summary.scalar('loss', val_metrics['loss'], step=epoch - 1)

                    # endregion Eval

                    # region Save ai_models
                    if self.best_loss < 0 or (val_metrics['loss']) < self.best_loss:
                        self.log_train.write_mylines(
                            '======================================================================\n')
                        self.log_train.write_mylines(
                            'Found better checkpoint! Saving to {}\n'.format(self.best_loss_checkpoint_dir))
                        self.log_train.write_mylines(
                            '======================================================================\n')
                        print('===========================================================================')
                        print('Found loss better checkpoint! Saving to {}'.format(self.best_loss_checkpoint_dir))
                        print('===========================================================================')
                        for f in os.listdir(self.best_loss_checkpoint_dir):
                            os.remove(os.path.join(self.best_loss_checkpoint_dir, f))

                        self.model.save_weights(os.path.join(self.best_loss_checkpoint_dir,
                                                             self.model_name + '-epoch-{}'.format(epoch)))

                        self.best_loss = val_metrics['loss']
                        self.bk_metric['best_loss'] = self.best_loss
                        bk_metric_file = open('{}/{}_bk_metric.txt'.format(self.log_dir, self.model_name), 'w')
                        json.dump(self.bk_metric, bk_metric_file)
                        bk_metric_file.close()
                        self.best_weights = self.model.get_weights()
                        self.wait = 0
                    # endregion
                    elif self.best_loss >= 0:
                        self.wait += 1
                        if self.wait > self.patience:
                            self.epoch_early_stopping = epoch
                            self.model.stop_training = True
                            self.log_train.write_mylines(
                                '======================================================================\n')
                            self.log_train.write_mylines(
                                'Restoring ai_models weights from best total_loss! Saving to {}\n'.format(
                                    self.saved_ae_model_dir))
                            self.log_train.write_mylines(
                                '======================================================================\n')
                            print('======================================================================\n')
                            print(
                                'Restoring ai_models weights from best total_loss! Saving to {}\n'.format(
                                    self.saved_ae_model_dir))
                            print('======================================================================\n')
                            self.model.set_weights(self.best_weights)
                            self.model.save(self.saved_ae_model_dir)
                            encoder = tf.keras.Model(self.model.layers[0].input, self.model.layers[1].output)
                            encoder.save(self.saved_encoder_model_dir)

                    sys.stdout.flush()
                    # endregion Save ai_models

                    with open(self.log_dir + '/{}_log.csv'.format(self.model_name), mode='a+') as report_file:
                        report_writer = csv.DictWriter(report_file, fieldnames=self.fieldnames)
                        report_writer.writerow(report_row)

            def on_train_end(self, logs=None):
                if self.epoch_early_stopping > 0:
                    self.log_train.write_mylines(
                        '======================================================================\n')
                    self.log_train.write_mylines(
                        'Early stopping! ai_models weights from the end of the best best_loss_eval={}\n'.format(
                            self.best_loss))
                    self.log_train.write_mylines(
                        '======================================================================\n')
                    log_train.write_mylines(
                        '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
                    print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
                    log_train.write_mylines(
                        '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

        def _preprocess_proto(example_proto, feature_len):
            """Read sample from protocol buffer."""
            encoding_scheme = {
                'sample': tf.io.FixedLenFeature(shape=[feature_len, ], dtype=tf.int64),
                'label': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            }
            proto = tf.io.parse_single_example(example_proto, encoding_scheme)
            sample = proto['sample']

            return sample, sample

        def _preprocess_proto_test(example_proto, feature_len):
            """Read sample from protocol buffer."""
            encoding_scheme = {
                'sample': tf.io.FixedLenFeature(shape=[feature_len, ], dtype=tf.int64),
                'label': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            }
            proto = tf.io.parse_single_example(example_proto, encoding_scheme)
            sample = proto['sample']
            label = proto['label']
            return sample, label

        with tf.device('/cpu:0'):
            # region Load training dataset
            train_filenames = self._get_tfrecord_filenames(train_directory, True)
            train_dataset = tf.data.TFRecordDataset(train_filenames)

            train_dataset = train_dataset.map(partial(_preprocess_proto,
                                                      feature_len=feature_len),
                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

            train_dataset = train_dataset.shuffle(buffer_size=8192)
            train_dataset = train_dataset.batch(batch_size)
            train_dataset = train_dataset.prefetch(batch_size * 5)
            # endregion

            # region Load test dataset
            val_filenames = self._get_tfrecord_filenames(test_directory, False)
            val_dataset = tf.data.TFRecordDataset(val_filenames)

            val_dataset = val_dataset.map(partial(_preprocess_proto,
                                                  feature_len=feature_len),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

            val_dataset = val_dataset.batch(batch_size)
            val_dataset = val_dataset.prefetch(batch_size * 5)
            # endregion
        # disable_eager_execution()
        sound_models = AutoencoderModels()
        train_model = getattr(sound_models, model_name)(feature_len,
                                                        feature_width,
                                                        feature_height,
                                                        num_filters,
                                                        kernel_size,
                                                        latent_dim,
                                                        decode_activation)

        train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                            loss=tf.keras.losses.MeanAbsoluteError())

        if resume_from is not None:
            begin_at_epoch = int(resume_from.split('-')[-1])
            log_train.write_mylines('Restoring checkpoint from {}\n'.format(resume_from))
            log_train.write_mylines('Beginning at epoch {}\n'.format(begin_at_epoch + 1))
            print('Restoring checkpoint from {}'.format(resume_from))
            print('Beginning at epoch {}'.format(begin_at_epoch + 1))
            train_model.load_weights(tf.train.latest_checkpoint(saved_encoder_model_dir)).expect_partial()
        else:
            begin_at_epoch = 0
            print('===================================================================================')
            print('WARNING: --resume_from checkpoint flag is not set. Training ai_models from scratch.')
            print('===================================================================================')

        if bk_metric is None:
            best_loss = -1
            bk_metric = dict()
            bk_metric['best_loss'] = -1
            bk_metric['stop_train'] = False
            bk_metric_file = open('{}/{}_bk_metric.txt'.format(log_dir, model_name), 'w')
            json.dump(bk_metric, bk_metric_file)
            bk_metric_file.close()
        else:
            best_loss = bk_metric['best_loss']

        tensorboard_dir = model_dir + '/logs'
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
            os.makedirs(tensorboard_dir + '/train')
            os.makedirs(tensorboard_dir + '/validation')

        log_callback = CustomCallback(

            model_name=model_name,
            log_train=log_train,
            stopped_epoch=begin_at_epoch + epoch_num,
            saved_encoder_model_dir=saved_encoder_model_dir,
            best_loss_checkpoint_dir=best_loss_checkpoint_dir,
            saved_ae_model_dir=saved_ae_model_dir,
            bk_metric=bk_metric,
            log_dir=log_dir,
            field_name=fieldnames,
            best_loss=best_loss,
            length_train=self._calc_num_steps(datastore_dict['train']['total_sample'], batch_size),
            length_valid=self._calc_num_steps(datastore_dict['test']['total_sample'], batch_size),
            valid_freq=valid_freq,
            patience=patience,
            tensorboard_dir=tensorboard_dir)

        with tf.device('/gpu:{}'.format(use_gpu_index if use_gpu_index >= 0 else 0)):
            train_model.fit(
                x=train_dataset,
                epochs=begin_at_epoch + epoch_num,
                verbose=0,
                callbacks=[log_callback],
                validation_data=val_dataset,
                validation_freq=[valid_freq * (x + 1) for x in range((begin_at_epoch + epoch_num) // valid_freq)],
                initial_epoch=begin_at_epoch)

        bk_metric['stop_train'] = True
        bk_metric_file = open('{}/{}_bk_metric.txt'.format(log_dir, model_name), 'w')
        json.dump(bk_metric, bk_metric_file)
        bk_metric_file.close()

        log_train.write_mylines('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        log_train.write_mylines('\nEnd : {}\n'.format(str(datetime.datetime.now())))
        log_train.write_mylines('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

        encoder = tf.keras.models.load_model(saved_encoder_model_dir)
        label = np.array(())
        signal = np.zeros((1, feature_len))
        eval_filenames = self._get_tfrecord_filenames(eval_directory, False)
        eval_filenames = tf.data.TFRecordDataset(eval_filenames)

        eval_filenames = eval_filenames.map(partial(_preprocess_proto_test,
                                                    feature_len=feature_len),
                                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        eval_filenames = eval_filenames.batch(batch_size)
        eval_filenames = eval_filenames.prefetch(batch_size * 5)

        for data in eval_filenames:
            signal = np.append(signal, data[0], axis=0)
            label = np.append(label, data[1])

        signal = signal[1:, :]
        z_mean = encoder.predict(signal)
        from sklearn.decomposition import PCA
        import time
        pca_2d = PCA(n_components=2)
        PCA_hidden_2d = pca_2d.fit_transform(z_mean)
        # Plot the principal components
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(111)
        p1 = ax1.scatter(PCA_hidden_2d[:, 0], PCA_hidden_2d[:, 1], c=label,
                         cmap='tab10')  # matplotlib.colors.ListedColormap(colors))
        plt.legend(handles=p1.legend_elements()[0], labels=datastore_dict['event_types'])
        img_name = log_dir + '/{}'.format(int(time.time()))
        fig.savefig(img_name + ".svg", format='svg', dpi=1200)
        return False


def unzip(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def covert_model2tflite(use_gpu_index, model_name_dir, datastore_file, checkpoint_dir, tflite_output_dir,
                        eval_directory, num_sample=None, memory_limit=1024):
    """

    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(use_gpu_index)

    import tensorflow as tf
    import tensorflow_model_optimization as tfmot
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)
    tf.autograph.set_verbosity(1)

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
        )
    else:
        my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')

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

    def representative_dataset_gen(dataset, sample_count):
        def generator():
            for features in dataset.take(sample_count):
                data_raw = features[0].numpy()
                yield [data_raw.astype(np.float32)]

        return generator

    def convert_int8_tflite(model, saved_model_int8_tflite_dir, saved_model_float32_tflite_dir, representative_dataset):
        """
        :param model:
        :param saved_model_int8_tflite_dir:
        :param saved_model_float32_tflite_dir:
        :param representative_dataset:
        :return:
        """

        # Convert the model to the TensorFlow Lite format without quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        model_float32_tflite = converter.convert()
        # Save the model to disk
        open(saved_model_float32_tflite_dir, "wb").write(model_float32_tflite)
        print('Converted to TF Lite float32 quantized model: %s; Size: %d KB.' %
              (saved_model_float32_tflite_dir, len(model_float32_tflite) / 1024))

        # Set the optimization flag.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Enforce integer only quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        # Provide a representative dataset to ensure we quantize correctly.
        converter.representative_dataset = representative_dataset
        model_tflite = converter.convert()
        # Save the model to disk
        open(saved_model_int8_tflite_dir, "wb").write(model_tflite)
        print('Converted to TF Lite int8 quantized model: %s; Size: %d KB.' %
              (saved_model_int8_tflite_dir, len(model_tflite) / 1024))

        return len(model_float32_tflite) / 1024, len(model_tflite) / 1024, model_tflite

    with open(datastore_file, 'r') as json_file:
        datastore_dict = json.load(json_file)

    feature_height = datastore_dict['feature_height']
    feature_width = datastore_dict['feature_width']
    feature_len = datastore_dict['feature_len']
    type_num = len(datastore_dict['event_types'])

    model_name_dir_path = model_name_dir.split('_')
    model_name = ""
    m = 0
    for m in range(len(model_name_dir_path)):
        if model_name_dir_path[m].isnumeric():
            break
        else:
            model_name += model_name_dir_path[m] + "_"

    model_name = model_name[:-1]
    latent_dim = int(model_name_dir_path[m])
    num_filters = np.asarray([int(i) for i in model_name_dir_path[m + 1].split('.')], dtype=int)
    kernel_size = np.asarray([int(i) for i in model_name_dir_path[m + 2].split('.')], dtype=int)
    decode_activation = model_name_dir_path[m + 3]

    if not os.path.exists(tflite_output_dir):
        os.makedirs(tflite_output_dir)

    MODEL_TFLITE = tflite_output_dir + '/{}_int8.tflite'.format(model_name)
    MODEL_NO_QUANT_TFLITE = tflite_output_dir + '/{}_float32.tflite'.format(model_name)
    MODEL_TFLITE_MICRO = tflite_output_dir + '/{}'.format(model_name)

    # covert_model = tf.keras.models.load_model(checkpoint_dir)
    sound_models = AutoencoderModels()
    covert_model = getattr(sound_models, model_name)(feature_len,
                                                     feature_width,
                                                     feature_height,
                                                     num_filters,
                                                     kernel_size,
                                                     latent_dim,
                                                     decode_activation)
    # covert_model.summary()
    covert_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    dataset = get_dataset(dir_path=eval_directory,
                          feature_len=feature_len,
                          class_num=type_num)

    if num_sample is None:
        sample_count = datastore_dict['test']['total_sample']
    else:
        sample_count = num_sample

    representative_dataset = representative_dataset_gen(dataset=dataset,
                                                        sample_count=sample_count)

    size_model_float32_tflite, size_model_int8_tflite, model_tflite = \
        convert_int8_tflite(model=covert_model,
                            saved_model_int8_tflite_dir=MODEL_TFLITE,
                            saved_model_float32_tflite_dir=MODEL_NO_QUANT_TFLITE,
                            representative_dataset=representative_dataset)

    from tensorflow.lite.python.util import convert_bytes_to_c_source
    source_text, header_text = convert_bytes_to_c_source(model_tflite, 'tfmodel')
    with open(MODEL_TFLITE_MICRO + '.h', 'w') as file:
        file.write(header_text)
    with open(MODEL_TFLITE_MICRO + '.cc', 'w') as file:
        file.write(source_text)

    tflite_dict = dict()
    tflite_dict["tflite_int8_size"] = size_model_int8_tflite
    tflite_dict["size_model_float32_tflite"] = size_model_float32_tflite
    tflite_dict["tflite_micro_dir"] = MODEL_TFLITE_MICRO
    des = os.path.dirname(eval_directory)
    log_tflite = open(des + '/tflite_info.txt', 'w')
    json.dump(tflite_dict, log_tflite)
    log_tflite.close()
    print(tflite_dict)
    return MODEL_TFLITE, MODEL_NO_QUANT_TFLITE, MODEL_TFLITE_MICRO


def main():
    MAIN_DIR = "20230321093220"
    MODEL_DIR = "D:/motor_failure_detection/{}".format(MAIN_DIR)

    start_time = time.time()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--epochs', type=int, default=1000)
    # parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--learning_rate', type=float, default=1e-3)
    # parser.add_argument('--model_name', type=str, default='auto_conv_tiny')
    # parser.add_argument('--ts', type=str, default='')
    # parser.add_argument('--decode_activation', type=str, default='none')
    # parser.add_argument('--kernel_size', type=str, default='3.3')
    # parser.add_argument('--num_filters', type=str, default='8.8.8')
    # parser.add_argument('--latent_dim', type=int, default=8)
    #
    # parser.add_argument('--model-dir', type=str, default=MODEL_DIR)
    #
    # args, _ = parser.parse_known_args()

    # batch_size = args.batch_size
    # model_dir = args.model_dir
    # max_epochs = args.epochs
    # model_name = args.model_name
    # str_num_filters = args.num_filters
    # str_kernel_size = args.kernel_size
    # latent_dim = args.latent_dim
    # ts = calendar.timegm(time.gmtime())
    # learning_rate = float(args.learning_rate)
    # decode_activation = args.decode_activation
    # num_filters = np.asarray([int(i) for i in str_num_filters.split('.')], dtype=int)
    # kernel_size = np.asarray([int(i) for i in str_kernel_size.split('.')], dtype=int)

    model_name = 'auto_conv_tiny'
    batch_size = 128
    max_epochs = 1000
    decode_activation = 'relu'
    learning_rate = 1e-3

    n_str_num_filters = ['8.8.8']
    n_str_kernel_size = ['3.3']
    n_latent_dim = [8]

    if not (os.path.isfile(MODEL_DIR + '/datastore.txt')):
        print('Not find file datastore.txt')
        return

    for str_num_filters, str_kernel_size, latent_dim in zip(n_str_num_filters, n_str_kernel_size, n_latent_dim):
        ts = calendar.timegm(time.gmtime())
        num_filters = np.asarray([int(i) for i in str_num_filters.split('.')], dtype=int)
        kernel_size = np.asarray([int(i) for i in str_kernel_size.split('.')], dtype=int)

        datastore_file = MODEL_DIR + '/datastore.txt'

        f = open(datastore_file, 'r')
        datastore_dict = json.load(f)

        data_directory = MODEL_DIR + "/data"
        output_tflite_dir = MODEL_DIR + '/output_tflite'

        train_directory = data_directory + '/train'
        test_directory = data_directory + '/test'
        eval_directory = data_directory + '/eval'
        print('model_dir: {}'.format(MODEL_DIR))
        print('train_directory: {}'.format(train_directory))
        print('test_directory: {}'.format(test_directory))

        print('Total TFrecords training files: {}'.format(len(glob(train_directory + '/*.tfrecord'))))
        print('Total TFrecords validation files: {}'.format(len(glob(test_directory + '/*.tfrecord'))))

        output_dir = MODEL_DIR + '/output'
        os.makedirs(output_dir, exist_ok=True)
        training = Training()
        model_name_dir = "{}_{}_{}_{}_{}_{}_{}_{}".format(model_name,
                                                          latent_dim,
                                                          str_num_filters,
                                                          str_kernel_size,
                                                          decode_activation,
                                                          batch_size,
                                                          learning_rate,
                                                          ts)

        model_dir = '{}/models/{}'.format(output_dir, model_name_dir)
        log_dir = '{}/logs/{}'.format(output_dir, model_name_dir)
        tflite_dir = '{}/output/{}'.format(output_tflite_dir, model_name_dir)

        print('model_dir: {}'.format(model_dir))

        for i in [model_dir, log_dir]:
            os.makedirs(i, exist_ok=True)

        print("decode_activation: {}".format(decode_activation))
        print('num_loops: {}'.format(latent_dim))
        print('num_filters: {}'.format(num_filters))
        print('kernel_size: {}'.format(kernel_size))
        process_train = multiprocessing.Process(target=training.train_vae, args=(0,
                                                                                 model_name,
                                                                                 latent_dim,
                                                                                 num_filters,
                                                                                 kernel_size,
                                                                                 log_dir,
                                                                                 model_dir,
                                                                                 datastore_dict,
                                                                                 None,
                                                                                 train_directory,
                                                                                 test_directory,
                                                                                 eval_directory,
                                                                                 batch_size,
                                                                                 learning_rate,
                                                                                 decode_activation,
                                                                                 3,
                                                                                 4,
                                                                                 max_epochs))
        process_train.start()
        process_train.join()

        print('training time: {}s'.format(time.time() - start_time))

        checkpoint_dir = model_dir + '/best_autoencoder_loss'

        if not os.path.exists(tflite_dir):
            os.makedirs(tflite_dir)

        print("datastore_file: {}".format(datastore_file))

        print("tflite_output_dir: {}".format(tflite_dir))

        model_tflite, model_no_quant_tflite, model_tflite_micro = \
            covert_model2tflite(use_gpu_index=0,
                                model_name_dir=model_name_dir,
                                datastore_file=datastore_file,
                                checkpoint_dir=checkpoint_dir,
                                tflite_output_dir=tflite_dir,
                                eval_directory=test_directory)


if __name__ == '__main__':
    main()
