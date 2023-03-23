import os
import wave

import numpy as np

PATH = "D:\sound_dataset\environment\motor_sound"
event_types = ['background2', 'motor_run2', 'motor_abnormal2']

sound_dir = "{}/datasets".format(PATH)
if not os.path.exists(sound_dir):
    os.makedirs(sound_dir)

for sound in event_types:
    files = os.listdir("{}/{}".format(PATH, sound))
    sound_type_dir = '{}\\{}'.format(sound_dir, sound)
    if not os.path.exists(sound_type_dir):
        os.makedirs(sound_type_dir)

    file_count = 0
    for file_name in files:
        if "wav" not in file_name:
            continue

        wav_obj = wave.open("{}\\{}\\{}".format(PATH, sound, file_name), 'rb')
        sample_freq = wav_obj.getframerate()
        n_samples = wav_obj.getnframes()
        t_audio = n_samples / sample_freq
        signal_wave = wav_obj.readframes(n_samples)
        waveform = np.frombuffer(signal_wave, dtype=np.int16)
        clip_sample = 15520
        index = np.arange(clip_sample)[None, :] + np.arange(0, n_samples - clip_sample, clip_sample)[:, None]
        data = waveform[index]
        for i in range(1, len(index)):
            wav = wave.open('{}\\{}.wav'.format(sound_type_dir, file_count), "wb")
            wav.setnchannels(wav_obj.getnchannels())
            wav.setsampwidth(wav_obj.getsampwidth())  # Sample width: 16 bits
            wav.setframerate(wav_obj.getframerate())  # Sample rate: 16kHz
            audio = (data[i]).astype("<H")
            wav.writeframes(audio.tobytes())
            wav.close()
            file_count += 1

