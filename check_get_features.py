import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from multiprocessing import Process, JoinableQueue, Lock
from sound_process import input_data
from sound_process import models
import json
import logging
import os
import warnings
import wave

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


MAIN_DIR = "20230301T120052"
MODEL_DIR = "D:/motor_failure_detection/{}".format(MAIN_DIR)
log_dir = "{}/output/models".format(MODEL_DIR)
model_dir_list = [name for name in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, name))]
# PATH = "D:/sound_dataset/environment/motor_sound"
PATH = "D:/sound_dataset/environment/motor_sound/clip_file"
event_types = ['background2',  'motor_run2', 'motor_abnormal2']
files = []
types = []
ind = {k: i for i, k in enumerate(event_types)}

for e in event_types:
    file_names = os.listdir("{}/{}".format(PATH, e))
    _files = ["{}/{}".format(PATH, e) + "/" + file_name for file_name in file_names if "wav" in file_name]
    _types = [ind[e] for file_name in file_names if "wav" in file_name]
    random_index = np.random.choice(len(_files), min(2000, len(_files)), replace=False)
    _files = np.asarray(_files)[random_index]
    _types = np.asarray(_types)[random_index]
    files += list(_files)
    types += list(_types)

for model_dir in model_dir_list:
    print(model_dir)
    output_debug_dir = "{}/{}".format(MAIN_DIR, model_dir)
    if not os.path.exists(output_debug_dir):
        os.makedirs(output_debug_dir)

    checkpoint_dir = "{}/output/models/{}/save_encoder_model".format(MODEL_DIR, model_dir)
    vae_checkpoint_dir = "{}/output/models/{}/saved_vae_model".format(MODEL_DIR, model_dir)
    datastore_file = MODEL_DIR + '/datastore.txt'
    f = open(datastore_file, 'r')
    datastore_dict = json.load(f)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(0)

    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)
    tf.autograph.set_verbosity(1)

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)]
        )
    else:
        my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')

    sess = tf.compat.v1.InteractiveSession()
    feature_len = datastore_dict['feature_len']
    encoder_model = tf.keras.models.load_model(checkpoint_dir)
    # vae_model = tf.keras.models.load_model(vae_checkpoint_dir)
    # encoder_model.summary()
    # Initialize the TFLite interpreter
    z_means = []
    z_types = []
    _mse = []
    lst_process_data = []
    from time import process_time
    sess = tf.compat.v1.InteractiveSession()
    model_settings = models.prepare_model_settings(
        0, datastore_dict['sample_rate'], 970, 30, 20,
        40, datastore_dict["preprocess"])
    audio_processor = input_data.AudioProcessor(None, None, 0, 0, '', 0, 0, model_settings, None)

    for data_path, data_type in zip(files, types):
        t1_start = process_time()
        results = audio_processor.get_features_for_wav(data_path, model_settings, sess)
        features = results[0]
        if datastore_dict["quantize"]:
            features_min, features_max = input_data.get_features_range(model_settings)
            features = np.asarray(np.round((255 * (features - features_min)) / (features_max - features_min)),
                                  dtype=np.int64)

        t1_stop = process_time()

        print("{} take time to process in seconds {}".format(data_path, t1_stop - t1_start))

    sess.close()