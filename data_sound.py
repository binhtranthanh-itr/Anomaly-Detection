import copy
import csv
import json
import os
import shutil
import sys
import wave, audioop
from datetime import datetime
from multiprocessing import Process, JoinableQueue, Lock
from os.path import dirname, basename
from random import shuffle
from numpy import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from sound_process import input_data
from sound_process import models

print('Tensorflow-version: {}'.format(tf.__version__))
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print('Physical Devices: GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print('+ Physical Devices: CPU')

S3_STUDY_DATA_BUCKET = 'btcy-eks-puma-customer-study-data'
AWS_REGION_NAME = os.environ.get('AWS_REGION_NAME', 'us-east-2')
MAX_NUM_IMG_SAVE = 100

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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


def np_to_tfrecords(sample_buffer, label_buffer, writer):
    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:
            raise ValueError("The input should be numpy ndarray. \
                               Instaed got {}".format(ndarray.dtype))

    try:
        assert isinstance(sample_buffer, np.ndarray)
        assert len(sample_buffer.shape) == 2  # If X has a higher rank,
        # it should be rshape before fed to this function.
        assert isinstance(label_buffer, np.ndarray) or label_buffer is None

        # load appropriate tf.train.Feature class depending on dtype
        dtype_feature_x = _dtype_feature(sample_buffer)
        if label_buffer is not None:
            assert sample_buffer.shape[0] == label_buffer.shape[0]
            assert len(label_buffer.shape) == 2
            dtype_feature_y = _dtype_feature(label_buffer)

        # iterate over each sample and serialize it as ProtoBuf.
        for idx in range(sample_buffer.shape[0]):
            x = sample_buffer[idx]
            d_feature = dict()
            d_feature['sample'] = dtype_feature_x(x)
            if label_buffer is not None:
                d_feature['label'] = dtype_feature_y(label_buffer[idx])

            features = tf.train.Features(feature=d_feature)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)

    except Exception as err:
        print('np_to_tfrecords [Error line: {}]: {}'.format(sys.exc_info()[-1].tb_lineno, err))


def initiate_process_parameters(res_db_dict, ranges):
    return_dict = dict()
    return_dict['process_res_db_dict'] = [0] * len(ranges)
    return_dict['process_lst_file_to_handle'] = [0] * len(ranges)

    for i in range(len(ranges)):
        return_dict['process_res_db_dict'][i] = copy.deepcopy(res_db_dict)
        return_dict['process_res_db_dict'][i]['test']['total_sample'] = 0
        return_dict['process_res_db_dict'][i]['train']['total_sample'] = 0
        for key in res_db_dict['event_types']:
            return_dict['process_res_db_dict'][i]['test'][key] = 0
            return_dict['process_res_db_dict'][i]['train'][key] = 0

        return_dict['process_lst_file_to_handle'] = dict()
        return_dict['process_lst_file_to_handle'][i] = list()

    return return_dict


def get_features(sess,
                 sample_rate,
                 clip_duration_ms,
                 window_size_ms,
                 window_stride_ms,
                 feature_bin_count,
                 quantize,
                 preprocess,
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
    audio_processor = input_data.AudioProcessor(None, None, 0, 0, '', 0, 0,
                                                model_settings, None)

    results = audio_processor.get_features_for_wav(input_wav, model_settings,
                                                   sess)
    features = results[0]
    if quantize:
        features_min, features_max = input_data.get_features_range(model_settings)
        features = np.asarray(np.round((255 * (features - features_min)) / (features_max - features_min)),
                              dtype=np.int64)

    return features


def process_sample(sess,
                   audio_processor,
                   model_settings,
                   use_gpu_index,
                   process_index,
                   lock,
                   num_augmentation,
                   data_path,
                   event_type,
                   ds_type,
                   res_db_dict,
                   writer,
                   output_directory,
                   save_image):
    dir_img_deb = output_directory + 'img/'
    try:
        if num_augmentation > 0 and \
                'train' in ds_type and \
                res_db_dict[ds_type]['augmentation_event'][event_type] < num_augmentation:

            x = random.randint(len(res_db_dict['background_noise']))
            background_file = res_db_dict['background_noise'][x]
            wav_obj = wave.open(background_file, 'rb')
            n_samples = wav_obj.getnframes()
            background_wave = wav_obj.readframes(n_samples)
            r = res_db_dict['list_ratio_augmentation'][random.randint(len(res_db_dict['list_ratio_augmentation']))]
            background_wave = audioop.mul(background_wave, wav_obj.getsampwidth(), r)
            background_waveform = np.frombuffer(background_wave, dtype=np.uint16)
            y = random.randint(n_samples - int((res_db_dict['clip_duration_ms'] / 1000) * res_db_dict['sample_rate']))
            background_waveform = background_waveform[y: y + int((res_db_dict['clip_duration_ms'] / 1000) *
                                                                 res_db_dict['sample_rate'])]
            wav_obj.close()
            wav_obj = wave.open(data_path, 'rb')
            n_samples = wav_obj.getnframes()
            signal_wave = wav_obj.readframes(n_samples)
            waveform = np.frombuffer(signal_wave, dtype=np.uint16)
            _waveform = waveform + background_waveform
            data_path_augmentation = data_path[:-4] + "_aug_{}_{}_{}.wav".format(basename(background_file)[:-4], y, r)
            wav = wave.open(data_path_augmentation, "wb")
            wav.setnchannels(wav_obj.getnchannels())  # Mono or Stereo
            wav.setsampwidth(wav_obj.getsampwidth())  # Sample width: 16 bits
            wav.setframerate(wav_obj.getframerate())  # Sample rate: 16kHz
            wav.writeframesraw(_waveform)
            wav.close()
            wav_obj.close()

            results = audio_processor.get_features_for_wav(data_path_augmentation, model_settings, sess)
            _features = results[0]
            if res_db_dict["quantize"]:
                features_min, features_max = input_data.get_features_range(model_settings)
                _features = np.asarray(np.round((255 * (_features - features_min)) / (features_max - features_min)),
                                       dtype=np.int64)

            features = _features.copy()

            ind = {k: i for i, k in enumerate(res_db_dict['event_types'])}
            if save_image is not None and save_image:
                wav_obj = wave.open(data_path_augmentation, 'rb')
                sample_freq = wav_obj.getframerate()
                n_samples = wav_obj.getnframes()
                signal_wave = wav_obj.readframes(n_samples)
                waveform = np.frombuffer(signal_wave, dtype=np.int16)

                sub_save_image = dir_img_deb + "/{}".format(event_type)
                if not os.path.exists(sub_save_image):
                    os.makedirs(sub_save_image)
                    file_count = 0
                else:
                    _, _, files = next(os.walk(sub_save_image))
                    file_count = len(files)

                if file_count < MAX_NUM_IMG_SAVE:
                    fig, axes = plt.subplots(2, figsize=(12, 8))
                    times = np.linspace(0, n_samples / sample_freq, num=n_samples)
                    axes[0].plot(times, waveform)
                    axes[0].set_title('Waveform {}'.format(event_type))

                    axes[1].imshow(features, interpolation='none')
                    axes[1].set_title('Spectrogram')
                    DEBUG_IMG = False
                    if not DEBUG_IMG:
                        img_name = sub_save_image + "/{}".format(basename(data_path_augmentation)[:-4])
                        fig.savefig(img_name + ".svg", format='svg', dpi=1200)
                        plt.close(fig)
                    else:
                        plt.show()

            feature_len = features.shape[0] * features.shape[1]
            res_db_dict[ds_type][event_type] += 1
            res_db_dict[ds_type]['augmentation_event'][event_type] += 1
            res_db_dict[ds_type]["total_sample"] += 1
            np_to_tfrecords(sample_buffer=np.reshape(features.flatten(), (-1, feature_len)),
                            label_buffer=np.reshape(np.asarray([ind[event_type]], dtype=np.int64), (-1, 1)),
                            writer=writer)

        results = audio_processor.get_features_for_wav(data_path, model_settings, sess)
        _features = results[0]
        if res_db_dict["quantize"]:
            features_min, features_max = input_data.get_features_range(model_settings)
            _features = np.asarray(np.round((255 * (_features - features_min)) / (features_max - features_min)),
                                   dtype=np.int64)

        features = _features.copy()
        ind = {k: i for i, k in enumerate(res_db_dict['event_types'])}
        if save_image is not None and save_image:
            wav_obj = wave.open(data_path, 'rb')
            sample_freq = wav_obj.getframerate()
            n_samples = wav_obj.getnframes()
            signal_wave = wav_obj.readframes(n_samples)
            waveform = np.frombuffer(signal_wave, dtype=np.int16)

            sub_save_image = dir_img_deb + "/{}".format(event_type)
            if not os.path.exists(sub_save_image):
                os.makedirs(sub_save_image)
                file_count = 0
            else:
                _, _, files = next(os.walk(sub_save_image))
                file_count = len(files)

            if file_count < MAX_NUM_IMG_SAVE:
                fig, axes = plt.subplots(2, figsize=(12, 8))
                times = np.linspace(0, n_samples / sample_freq, num=n_samples)
                axes[0].plot(times, waveform)
                axes[0].set_title('Waveform {}'.format(event_type))
                axes[1].imshow(features, interpolation='none')
                DEBUG_IMG = False
                if not DEBUG_IMG:
                    img_name = sub_save_image + "/{}".format(basename(data_path)[:-4])
                    fig.savefig(img_name + ".svg", format='svg', dpi=1200)
                    plt.close(fig)
                else:
                    plt.show()

        feature_len = features.shape[0] * features.shape[1]
        res_db_dict['feature_len'] = feature_len
        res_db_dict['feature_width'] = features.shape[0]
        res_db_dict['feature_height'] = features.shape[1]
        res_db_dict[ds_type][event_type] += 1
        res_db_dict[ds_type]["total_sample"] += 1
        np_to_tfrecords(sample_buffer=np.reshape(features.flatten(), (-1, feature_len)),
                        label_buffer=np.reshape(np.asarray([ind[event_type]], dtype=np.int64), (-1, 1)),
                        writer=writer)

    except Exception as err:
        print('_process_sample [Error line: {}]: {} at {}'.format(sys.exc_info()[-1].tb_lineno, err, data_path))

    return res_db_dict


def process_event_batch(use_gpu_index,
                        queue,
                        lock,
                        process_index,
                        ranges,
                        res_db_dict,
                        all_data_path,
                        all_event_type,
                        output_directory,
                        num_shards,
                        total_shards,
                        save_image):
    num_processes = len(ranges)
    assert not num_shards % num_processes
    num_shards_per_batch = int(num_shards / num_processes)
    shard_ranges = np.linspace(ranges[process_index][0], ranges[process_index][1], num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[process_index][1] - ranges[process_index][0]
    # Initial parameter for each process

    counter = 0
    ds_type = 'train' if 'train' in basename(dirname(output_directory)) else 'test'
    for s in range(num_shards_per_batch):

        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = process_index * num_shards_per_batch + s
        output_filename = '%s_%.5d-of-%.5d.tfrecord' % (ds_type,
                                                        res_db_dict['previous_shards'][ds_type] + shard + 1,
                                                        total_shards)
        output_file = os.path.join(output_directory, output_filename)
        writer = tf.io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)

        num_augmentation = int(len(np.asarray(np.asarray(all_event_type)[files_in_shard])) *
                               res_db_dict["ratio_augmentation"])

        # Start a new TensorFlow session.
        sess = tf.compat.v1.InteractiveSession()
        model_settings = models.prepare_model_settings(
            0, res_db_dict['sample_rate'], res_db_dict['clip_duration_ms'], 30, 20,
            40, res_db_dict["preprocess"])
        audio_processor = input_data.AudioProcessor(None, None, 0, 0, '', 0, 0, model_settings, None)

        for i in files_in_shard:
            data_path = all_data_path[i]
            event_type = all_event_type[i]
            try:
                res_db_dict = process_sample(sess=sess,
                                             audio_processor=audio_processor,
                                             model_settings=model_settings,
                                             use_gpu_index=use_gpu_index,
                                             process_index=process_index,
                                             lock=lock,
                                             num_augmentation=num_augmentation,
                                             data_path=data_path,
                                             event_type=event_type,
                                             ds_type=ds_type,
                                             res_db_dict=res_db_dict,
                                             writer=writer,
                                             output_directory=output_directory,
                                             save_image=save_image)

            except Exception as e:
                lock.acquire()
                try:
                    print(e)
                    print('SKIPPED: Unexpected error while decoding %s.' % data_path)
                finally:
                    lock.release()
                continue

            shard_counter += 1
            counter += 1
            if not counter % 1000:
                lock.acquire()
                try:
                    print('%s [processor %d]: Processed %d of %d files in processing batch.' %
                          (datetime.now(), process_index, counter, num_files_in_thread))
                    sys.stdout.flush()
                finally:
                    lock.release()

        writer.close()
        lock.acquire()
        try:
            print('%s [processor %d]: Wrote %d files to %s' %
                  (datetime.now(), process_index, shard_counter, output_file))

            sys.stdout.flush()
        finally:
            lock.release()

        shard_counter = 0

    queue.put({'process_res_db_dict': res_db_dict})
    lock.acquire()
    try:
        print('%s [processor %d]: Wrote %d files to %d shards.' %
              (datetime.now(), process_index, counter, num_files_in_thread))
        sys.stdout.flush()
    finally:
        lock.release()

    queue.task_done()


def build_tfrecord(use_gpu_index,
                   db_process_info,
                   total_shards,
                   datastore_dict,
                   output_directory,
                   save_image):
    num_processes = db_process_info['processors']
    num_shards = db_process_info['shards']
    all_data_path = db_process_info['data_path']
    all_event_type = db_process_info['event_type']
    spacing = np.linspace(0, len(all_data_path), num_processes + 1).astype(np.int64)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a processor for each batch.
    # print('Launching %d processors for spacings: %s' % (num_processes, ranges))
    sys.stdout.flush()
    # Create a mechanism for monitoring when all processors are finished.
    coord = tf.train.Coordinator()

    # Initiate parameter for each process
    process_data_dict = initiate_process_parameters(datastore_dict, ranges)
    process_res_db_dict = process_data_dict['process_res_db_dict']
    processors = list()
    process_queue = [list() for _ in range(num_processes)]
    process_lock = [list() for _ in range(num_processes)]

    dir_img_deb = output_directory + 'img/'
    if not os.path.exists(dir_img_deb) and save_image:
        os.makedirs(dir_img_deb)

    for process_index in range(len(ranges)):
        process_queue[process_index] = JoinableQueue()
        process_lock[process_index] = Lock()
        args = (use_gpu_index,
                process_queue[process_index],
                process_lock[process_index],
                process_index,
                ranges,
                process_res_db_dict[process_index],
                all_data_path,
                all_event_type,
                output_directory,
                num_shards,
                total_shards,
                save_image)
        t = Process(target=process_event_batch, args=args)
        t.start()
        processors.append(t)

    # Get output of processes
    for process_index in range(len(ranges)):
        process_returned_data = process_queue[process_index].get()
        process_res_db_dict[process_index] = process_returned_data['process_res_db_dict']
        processors[process_index].terminate()

    # Wait for all the processors to terminate.
    coord.join(processors)

    # Concatenate processes returned output !!!
    ds_type = 'train' if 'train' in basename(dirname(output_directory)) else 'test'
    datastore_dict['previous_shards'][ds_type] += num_shards

    datastore_dict[ds_type]['total_sample'] += int(
        np.asarray([t[ds_type]['total_sample'] for t in process_res_db_dict]).sum())

    for key in datastore_dict['event_types']:
        datastore_dict[ds_type][key] += int(np.asarray([t[ds_type][key] for t in process_res_db_dict]).sum())
        datastore_dict[ds_type]['augmentation_event'][key] += \
            int(np.asarray([t[ds_type]['augmentation_event'][key] for t in process_res_db_dict]).sum())

    datastore_dict["feature_len"] = process_res_db_dict[-1]["feature_len"]
    datastore_dict["feature_width"] = process_res_db_dict[-1]["feature_width"]
    datastore_dict["feature_height"] = process_res_db_dict[-1]["feature_height"]
    sys.stdout.flush()
    return datastore_dict


def main():
    print("--- Create tfrecord ---")
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y%m%d%H%M%S")
    print("date and time:", date_time)

    MAIN_DIR = date_time
    AUDIO_PATH = "D:/sound_dataset/environment/motor_sound/datasets"
    DATA_MODEL_DIR = "D:/motor_failure_detection/{}/data".format(MAIN_DIR)
    BACKGROUND_PATH = "D:/sound_dataset/environment/motor_sound/_background_voice_"
    RATIO = 0.8
    event_types = ['background2', 'motor_run2', 'motor_abnormal2']

    if not os.path.exists(DATA_MODEL_DIR):
        os.makedirs(DATA_MODEL_DIR)

    print(DATA_MODEL_DIR)
    datastore_dict = dict()
    datastore_dict['event_types'] = event_types
    datastore_dict['audio_path'] = AUDIO_PATH

    datastore_dict['ratio_augmentation'] = 0.5
    datastore_dict['list_ratio_augmentation'] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    datastore_dict['background_noise'] = [BACKGROUND_PATH + '/' + f for f in os.listdir(BACKGROUND_PATH) if 'wav' in f]

    datastore_dict['sample_rate'] = 16000
    datastore_dict['clip_duration_ms'] = 970
    datastore_dict['window_size_ms'] = 30
    datastore_dict['window_stride_ms'] = 20
    datastore_dict['feature_bin_count'] = 40
    datastore_dict['quantize'] = True
    datastore_dict['preprocess'] = 'micro'
    datastore_dict['feature_len'] = 0
    datastore_dict['feature_width'] = 0
    datastore_dict['feature_height'] = 0
    datastore_dict['train'] = dict()
    datastore_dict['test'] = dict()
    datastore_dict['eval'] = dict()
    datastore_dict["total_event"] = dict()
    datastore_dict['train']['augmentation_event'] = dict()
    datastore_dict['test']['augmentation_event'] = dict()
    datastore_dict['eval']['augmentation_event'] = dict()

    datastore_dict['train_event'] = dict()
    datastore_dict['test_event'] = dict()
    datastore_dict['eval_event'] = dict()
    datastore_dict['train']['total_sample'] = 0
    datastore_dict['test']['total_sample'] = 0
    datastore_dict['eval']['total_sample'] = 0

    datastore_dict['previous_shards'] = dict()
    if not os.path.exists(DATA_MODEL_DIR + '/finish.txt'):
        fstatus = open(DATA_MODEL_DIR + '/start.txt', 'w')
        fstatus.writelines(str(datetime.now()))
        fstatus.close()
        ds_file = dict()
        ds_file['train'] = {'data_path': [], 'event_type': []}
        ds_file['test'] = {'data_path': [], 'event_type': []}
        ds_file['eval'] = {'data_path': [], 'event_type': []}
        csv_trainfile = open("{}/train-data.csv".format(DATA_MODEL_DIR), 'w')
        headers = ["data_path", "event_type"]
        train_writer = csv.DictWriter(csv_trainfile, fieldnames=headers)
        train_writer.writeheader()

        csv_testfile = open("{}/test-data.csv".format(DATA_MODEL_DIR), 'w')
        test_writer = csv.DictWriter(csv_testfile, fieldnames=headers)
        test_writer.writeheader()

        csv_evalfile = open("{}/eval-data.csv".format(DATA_MODEL_DIR), 'w')
        eval_writer = csv.DictWriter(csv_evalfile, fieldnames=headers)
        eval_writer.writeheader()

        for e in event_types:
            aug_files = [_f for _f in os.listdir("{}/{}".format(AUDIO_PATH, e)) if 'aug' in _f]
            for _f in aug_files:
                os.remove("{}/{}/{}".format(AUDIO_PATH, e, _f))

            files = [_f for _f in os.listdir("{}/{}".format(AUDIO_PATH, e)) if 'aug' not in _f and 'wav' in _f]
            shuffle(files)
            datastore_dict["total_event"][e] = len(files)

            datastore_dict["train_event"][e] = 0
            datastore_dict["test_event"][e] = 0
            datastore_dict["eval_event"][e] = 0

            datastore_dict['train']['augmentation_event'][e] = 0
            datastore_dict['test']['augmentation_event'][e] = 0
            datastore_dict['eval']['augmentation_event'][e] = 0

            datastore_dict["train"][e] = 0
            datastore_dict["test"][e] = 0
            datastore_dict["eval"][e] = 0
            for _f in files:
                if "wav" not in _f:
                    continue

                f = "{}/{}/{}".format(AUDIO_PATH, e, _f)
                if 'abnormal' not in e and datastore_dict["train_event"][e] < int(RATIO * datastore_dict["total_event"][e]):
                    ds_file['train']["data_path"].append(f)
                    ds_file['train']["event_type"].append(e)
                    datastore_dict["train_event"][e] += 1

                    r = {key: value for key, value in zip(headers, [f, e])}
                    train_writer.writerow(r)
                elif 'abnormal' not in e:
                    ds_file['test']["data_path"].append(f)
                    ds_file['test']["event_type"].append(e)
                    datastore_dict["test_event"][e] += 1
                    r = {key: value for key, value in zip(headers, [f, e])}
                    test_writer.writerow(r)

                    ds_file['eval']["data_path"].append(f)
                    ds_file['eval']["event_type"].append(e)
                    datastore_dict["eval_event"][e] += 1
                    r = {key: value for key, value in zip(headers, [f, e])}
                    eval_writer.writerow(r)
                else:
                    ds_file['eval']["data_path"].append(f)
                    ds_file['eval']["event_type"].append(e)
                    datastore_dict["eval_event"][e] += 1
                    r = {key: value for key, value in zip(headers, [f, e])}
                    eval_writer.writerow(r)

        csv_trainfile.close()
        csv_testfile.close()
        csv_evalfile.close()
        print(datastore_dict)

        for t in ['train', 'test', 'eval']:
            datastore_dict['previous_shards'][t] = 0
            out_dir = '{}/{}/'.format(DATA_MODEL_DIR, t)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            else:
                shutil.rmtree(out_dir)
                os.makedirs(out_dir)

            db_process_info = dict()
            try:
                all_data_path = ds_file[t]["data_path"].copy()
                num_processes, num_shards = cal_num_process_and_num_shard(all_data_path, 4, 4)
                db_process_info["data_path"] = ds_file[t]["data_path"]
                db_process_info["event_type"] = ds_file[t]["event_type"]
                db_process_info['processors'] = num_processes
                db_process_info['shards'] = num_shards
                total_shards = np.asarray([db_process_info['shards']]).sum()

                datastore_dict = build_tfrecord(use_gpu_index=0,
                                                db_process_info=db_process_info,
                                                total_shards=total_shards,
                                                datastore_dict=datastore_dict,
                                                output_directory=out_dir,
                                                save_image=True)

                print('num_{}_samples = {}'.format(t, datastore_dict[t]['total_sample']))

            except Exception as error:
                print('[Error Line: {}]: {}'.format(sys.exc_info()[-1].tb_lineno, error))

        log_datastore = open(dirname(DATA_MODEL_DIR) + '/datastore.txt', 'w')
        json.dump(datastore_dict, log_datastore)
        log_datastore.close()
        fstatus = open(DATA_MODEL_DIR + '/finish.txt', 'w')
        fstatus.writelines(str(datetime.now()))
        fstatus.close()
        print(datastore_dict)


if __name__ == '__main__':
    main()
