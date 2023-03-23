import json
import os
from functools import partial
from glob import glob
from os.path import basename, dirname
from random import shuffle
from tiny_ae_models import create_ae2_model, create_ae3_model
import numpy as np


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

    if not os.path.exists(tflite_output_dir):
        os.makedirs(tflite_output_dir)
    # else:
    #     shutil.rmtree(tflite_output_dir)
    #     os.makedirs(tflite_output_dir)

    MODEL_TFLITE = tflite_output_dir + '/{}_int8.tflite'.format(model_name)
    MODEL_NO_QUANT_TFLITE = tflite_output_dir + '/{}_float32.tflite'.format(model_name)
    MODEL_TFLITE_MICRO = tflite_output_dir + '/{}'.format(model_name)

    # covert_model = tf.keras.models.load_model(checkpoint_dir)
    covert_model = create_ae3_model(feature_len,
                                   feature_width,
                                   feature_height,
                                   8)
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
    des = dirname(eval_directory)
    log_tflite = open(des + '/tflite_info.txt', 'w')
    json.dump(tflite_dict, log_tflite)
    log_tflite.close()
    print(tflite_dict)
    return MODEL_TFLITE, MODEL_NO_QUANT_TFLITE, MODEL_TFLITE_MICRO


def main():
    MAIN_DIR = "20230321093220"
    MODEL_DIR = "D:/motor_failure_detection/{}".format(MAIN_DIR)
    log_dir = "{}/output/models".format(MODEL_DIR)
    representative = "{}/data/test".format(MODEL_DIR)
    model_dir_list = [name for name in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, name))]
    for model_dir in model_dir_list:
        print(model_dir)

        # if 'ae3_8_8_0_128_0.001_0.5_1679289972' != model_dir:
        #     continue

        # model_dir = "beat_net_tiny_3_8.16.24.32_0_32_0.0001_0.5_1671106958"
        checkpoint_dir = "{}/output/models/{}/best_ae_loss".format(MODEL_DIR, model_dir)
        datastore_file = MODEL_DIR + '/datastore.txt'
        output_tflite_dir = "{}/output_tflite/{}".format(MODEL_DIR, model_dir)
        if not os.path.exists(output_tflite_dir):
            os.makedirs(output_tflite_dir)

        tflite_output_dir = output_tflite_dir + "/output"
        if not os.path.exists(tflite_output_dir):
            os.makedirs(tflite_output_dir)

        print("datastore_file: {}".format(datastore_file))

        print("tflite_output_dir: {}".format(tflite_output_dir))

        model_tflite, model_no_quant_tflite, model_tflite_micro = \
            covert_model2tflite(use_gpu_index=0,
                                model_name_dir=model_dir,
                                datastore_file=datastore_file,
                                checkpoint_dir=checkpoint_dir,
                                tflite_output_dir=tflite_output_dir,
                                eval_directory=representative)


if __name__ == '__main__':
    main()
