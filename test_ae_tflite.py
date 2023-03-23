import json
import logging
import os
import warnings
import time

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from multiprocessing import Process, JoinableQueue, Lock
from sound_process import input_data
from sound_process import models

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


def get_features(sess, sample_rate, clip_duration_ms, window_size_ms,
                 window_stride_ms, feature_bin_count, quantize, preprocess,
                 input_wav):
    """Converts an audio file into its corresponding feature map.

    Args:
      sample_rate: Expected sample rate of the wavs.
      clip_duration_ms: Expected duration in milliseconds of the wavs.
      window_size_ms: How long each spectrogram timeslice is.
      window_stride_ms: How far to move in time between spectrogram timeslices.
      feature_bin_count: How many bins to use for the feature fingerprint.
      quantize: Whether to train the model for eight-bit deployment.
      preprocess: Spectrogram processing mode; "mfcc", "average" or "micro".
      input_wav: Path to the audio WAV file to read.
    """

    model_settings = models.prepare_model_settings(
        0, sample_rate, clip_duration_ms, window_size_ms, window_stride_ms,
        feature_bin_count, preprocess)
    audio_processor = input_data.AudioProcessor(None, None, 0, 0, '', 0, 0, model_settings, None)

    results = audio_processor.get_features_for_wav(input_wav, model_settings, sess)
    features = results[0]
    if quantize:
        features_min, features_max = input_data.get_features_range(model_settings)
        features = np.asarray(np.round((255 * (features - features_min)) / (features_max - features_min)),
                              dtype=np.int64)
    return features


def cal_num_process_and_num_shard(files, org_num_processes, org_num_shards):
    num_processes, num_shards = 0, 0
    if len(files) >= org_num_shards:
        num_processes = org_num_processes
        num_shards = org_num_shards
    else:
        for n_threads in reversed(range(org_num_processes)):
            if len(files) // n_threads >= 1:
                num_processes = n_threads
                num_shards = (len(files) // n_threads) * n_threads
                break

    return num_processes, num_shards


def test_process(use_gpu_index,
                 ds_file_testing,
                 checkpoint_dir,
                 tflite_output_dir,
                 datastore_dict,
                 num_processes,
                 output_debug_dir):
    # Create a mechanism for monitoring when all processors are finished.
    coord = tf.train.Coordinator()

    spacing = np.linspace(0, len(ds_file_testing), num_processes + 1).astype(np.int64)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    input_sig = []
    output_sig = []

    input_sig_int8 = []
    output_sig_int8 = []

    processors = list()
    process_queue = [list() for _ in range(num_processes)]
    process_lock = [list() for _ in range(num_processes)]
    for process_index in range(len(ranges)):
        process_queue[process_index] = JoinableQueue()
        process_lock[process_index] = Lock()

        args = (use_gpu_index,
                process_index,
                process_queue[process_index],
                process_lock[process_index],
                ds_file_testing[ranges[process_index][0]: ranges[process_index][1] + 1],
                checkpoint_dir,
                tflite_output_dir,
                datastore_dict,
                512,
                output_debug_dir)

        t = Process(target=test_vae, args=args)
        t.start()
        processors.append(t)

    # Get output of processes
    for process_index in range(len(ranges)):
        process_returned_data = process_queue[process_index].get()
        if len(input_sig) == 0:
            input_sig = process_returned_data['input_sig']
            output_sig = process_returned_data['output_sig']
            input_sig_int8 = process_returned_data['input_sig_int8']
            output_sig_int8 = process_returned_data['output_sig_int8']
        else:
            input_sig = np.concatenate((input_sig, process_returned_data['input_sig']), axis=0)
            output_sig = np.concatenate((output_sig, process_returned_data['output_sig']), axis=0)
            input_sig_int8 = np.concatenate((input_sig_int8, process_returned_data['input_sig_int8']), axis=0)
            output_sig_int8 = np.concatenate((output_sig_int8, process_returned_data['output_sig_int8']), axis=0)

        processors[process_index].terminate()

    # Wait for all the processors to terminate.
    coord.join(processors)
    return input_sig, output_sig, input_sig_int8, output_sig_int8


def test_vae(use_gpu_index,
             process_index,
             process_queue,
             process_lock,
             ds_file_testing,
             checkpoint_dir,
             tflite_output_dir,
             datastore_dict,
             memory=1024,
             dir_image=None):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(use_gpu_index)

    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)
    tf.autograph.set_verbosity(1)

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)]
        )
    else:
        my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')

    sess = tf.compat.v1.InteractiveSession()
    model_settings = models.prepare_model_settings(
        0, datastore_dict['sample_rate'], datastore_dict['clip_duration_ms'], 30, 20,
        40, datastore_dict["preprocess"])
    audio_processor = input_data.AudioProcessor(None, None, 0, 0, '', 0, 0, model_settings, None)

    feature_len = datastore_dict['feature_len']
    test_model = tf.keras.models.load_model(checkpoint_dir)

    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_output_dir)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_shape = input_details['shape']
    # If required, quantize the input layer (from float to integer)
    input_scale, input_zero_point = input_details["quantization"]
    print('input_scale= {}'.format(input_scale))
    print('input_zero_point= {}'.format(input_zero_point))
    print("output_shape: {}".format(output_details['shape']))

    input_sig = []
    output_sig = []
    input_sig_int8 = []
    output_sig_int8 = []
    from time import process_time
    for data_path in ds_file_testing:
        results = audio_processor.get_features_for_wav(data_path, model_settings, sess)
        _features = results[0]
        if datastore_dict["quantize"]:
            features_min, features_max = input_data.get_features_range(model_settings)
            _features = np.asarray(np.round((255 * (_features - features_min)) / (features_max - features_min)),
                                   dtype=np.int64)

        features = _features.copy()
        process_data = np.reshape(features, (-1, feature_len))

        if (input_scale, input_zero_point) != (0.0, 0):
            process_data_int8 = process_data / input_scale + input_zero_point
            process_data_int8 = process_data_int8.astype(input_details["dtype"])
        else:
            process_data_int8 = process_data.copy()

        interpreter.set_tensor(input_details['index'], np.reshape(process_data_int8, input_shape))
        interpreter.invoke()

        decoded_sig_int8 = interpreter.get_tensor(output_details['index'])
        decoded_sig = test_model.predict(process_data)

        if len(input_sig) == 0:
            input_sig = process_data.copy()
            input_sig_int8 = process_data_int8.copy()
            output_sig = decoded_sig.copy()
            output_sig_int8 = decoded_sig_int8.copy()
        else:
            input_sig = np.concatenate((input_sig, process_data), axis=0)
            input_sig_int8 = np.concatenate((input_sig_int8, process_data_int8), axis=0)
            output_sig = np.concatenate((output_sig, decoded_sig), axis=0)
            output_sig_int8 = np.concatenate((output_sig_int8, decoded_sig_int8), axis=0)

    # Stop the stopwatch / counter
    sess.close()
    process_queue.put({'input_sig': input_sig,
                       'output_sig': output_sig,
                       'input_sig_int8': input_sig_int8,
                       'output_sig_int8': output_sig_int8,
                       })
    process_lock.acquire()
    process_queue.task_done()


def main():
    MAIN_DIR = "20230321093220"
    MODEL_DIR = "D:/motor_failure_detection/{}".format(MAIN_DIR)
    log_dir = "{}/output/models".format(MODEL_DIR)
    model_dir_list = [name for name in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, name))]
    # PATH = "D:/sound_dataset/environment/motor_sound"
    PATH = "D:/sound_dataset/environment/motor_sound/datasets"
    files = []

    for e in ['background2', 'motor_run2']:
        file_names = os.listdir("{}/{}".format(PATH, e))
        _files = ["{}/{}".format(PATH, e) + "/" + file_name for file_name in file_names
                  if "wav" in file_name and 'aug' not in file_names]
        files += list(_files)

    file_names = os.listdir("{}/{}".format(PATH, 'motor_abnormal2'))
    files_abnormal = ["{}/{}".format(PATH, 'motor_abnormal2') + "/" + file_name for file_name in file_names
                      if "wav" in file_names]

    for model_dir in model_dir_list:
        print(model_dir)
        if 'auto_conv_tiny_8_8.8.8_3.3_relu_128_0.001_1679466516' != model_dir:
            continue

        output_debug_dir = "{}/{}".format(MAIN_DIR, model_dir)
        if not os.path.exists(output_debug_dir):
            os.makedirs(output_debug_dir)

        model_name_dir_path = model_dir.split('_')
        model_name = ""
        m = 0
        for m in range(len(model_name_dir_path)):
            if model_name_dir_path[m].isnumeric():
                break
            else:
                model_name += model_name_dir_path[m] + "_"

        model_name = model_name[:-1]

        checkpoint_dir = "{}/output/models/{}/saved_autoencoder_model".format(MODEL_DIR, model_dir)
        tflite_output_dir = "{}/output_tflite/output/{}/{}_int8.tflite".format(MODEL_DIR, model_dir, model_name)
        datastore_file = MODEL_DIR + '/datastore.txt'
        f = open(datastore_file, 'r')
        datastore_dict = json.load(f)

        random_index = np.random.choice(len(files), len(files), replace=False)
        files = np.asarray(files)[random_index]

        input_sig, output_sig, input_sig_int8, output_sig_int8 = test_process(0,
                                                                              files,
                                                                              checkpoint_dir,
                                                                              tflite_output_dir,
                                                                              datastore_dict,
                                                                              4,
                                                                              output_debug_dir)

        input_sig2, output_sig2, input_sig_int82, output_sig_int82 = test_process(0,
                                                                                  files_abnormal,
                                                                                  checkpoint_dir,
                                                                                  tflite_output_dir,
                                                                                  datastore_dict,
                                                                                  4,
                                                                                  output_debug_dir)

        # reconstructions = autoencoder.predict(normal_train_data)

        # Get the mean absolute error between actual and reconstruction/prediction
        prediction_loss = np.mean(np.abs(input_sig_int8 - output_sig_int8), axis=-1)
        prediction_abnormal_loss = np.mean(np.abs(input_sig_int82 - output_sig_int82), axis=-1)
        # Check the prediction loss threshold for 2% of outliers
        percent = 96
        loss_threshold = np.percentile(prediction_loss, percent)
        print("Threshold: ", loss_threshold)
        print(f'The prediction loss threshold for {percent}% of outliers is {loss_threshold:.2f}')
        # Visualize the threshold
        fig = plt.figure(figsize=(12, 10))
        sns.histplot(prediction_loss, bins=200, alpha=0.5)
        sns.histplot(prediction_abnormal_loss, bins=200, alpha=0.5)
        plt.axvline(x=loss_threshold, color='orange')
        plt.xlabel("Prediction loss")
        plt.title(f'The prediction loss threshold for {percent}% of outliers is {loss_threshold:.2f}')
        DEBUG_IMG = False
        if not DEBUG_IMG:
            img_name = '{}/{}_prediction_loss_threshold_{}'.format(output_debug_dir, int(time.time()),
                                                                   f'{loss_threshold:.2f}')
            fig.savefig(img_name + ".svg", format='svg', dpi=1200)
            plt.close(fig)
        else:
            plt.show()


if __name__ == '__main__':
    main()
