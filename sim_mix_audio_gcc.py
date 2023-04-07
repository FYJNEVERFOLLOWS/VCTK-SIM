import json
import multiprocessing as mp
import os
import shutil
import apkit
import numpy as np
import soundfile as sf
import wave
from scipy.io import wavfile
import pickle

import tools

np.set_printoptions(threshold=np.inf)
MAX_WORKERS = max(mp.cpu_count() - 5, 10)

_FREQ_MAX = 8000
_FREQ_MIN = 100
SEG_LEN = 8192
SEG_HOP = 4096

version = 6
source_num = 3
single_source_gt_folder = '/Work21/2021/fuyanjie/exp_data/sim_audio_vctk/vctk_gt_frame'

train_test_val_flag = f'test_{version}'
# train_test_val_flag = '50roomsA'
json_folder = f'/Work21/2021/fuyanjie/exp_data/sim_audio_vctk/json_{source_num}sources_2000_{train_test_val_flag}'
output_folder = f'/Work21/2021/fuyanjie/exp_data/sim_audio_vctk/mixed_{source_num}sources_audio_{train_test_val_flag}'
data_frame_path = f"/Work21/2021/fuyanjie/exp_data/sim_audio_vctk/sim_{source_num}sources_{train_test_val_flag}_gcc" # 每帧的特征和标签

# train_test_val_flag = '50rooms_2W_A'
# json_folder = f'/Work21/2021/fuyanjie/exp_data/sim_audio_vctk/json_3sources_20000_50roomsA'
# output_folder = f'/Work21/2021/fuyanjie/exp_data/sim_audio_vctk/mixed_3nosaudio_{train_test_val_flag}'
# data_frame_path = f"/Work21/2021/fuyanjie/exp_data/sim_audio_vctk/sim_3sources_{train_test_val_flag}_gcc" # 每帧的特征和标签

print(f"json_folder:\n{json_folder}", flush=True)
print(f"output_folder:\n{output_folder}", flush=True)
print(f"data_frame_path:\n{data_frame_path}", flush=True)


def audioread(path, segment, fs=48000):
    sr, wave_data = wavfile.read(path) # default dtype: int16
    # wave_data, sr = sf.read(path) # default dtype: float64
    # print(f'wave_data.shape {wave_data.shape} sr {sr}', flush=True)
    if sr != fs:
        # wave_data = librosa.resample(wave_data, sr, fs)
        raise ValueError("input wav file samplerate is not {}".format(fs))
    return wave_data[:int(fs * segment)]

def activelev(data):
    eps = np.finfo(np.float32).eps
    max_val = (1. + eps) / (np.std(data) + eps)
    data = data * max_val
    return data

# make directory of train/test data
if os.path.exists(data_frame_path):
    print(f'{data_frame_path} already exists!')
    shutil.rmtree(data_frame_path)
    print(f'{data_frame_path} removed')
os.makedirs(data_frame_path)

# make directory of output audio
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    print(f'{output_folder} already exists!')
    shutil.rmtree(output_folder)
    print(f'{output_folder} removed')
for num_of_sources in range(1, 5):
    os.makedirs(output_folder + os.sep + f'SourceNum-{num_of_sources}')

# 从 json folder 中读取不同声源并混合
file_path_lst = os.listdir(json_folder)[:2500]

cnt_segs = 0 # gt_frame actually means gt_segment 
cnt_source_num_0 = 0
cnt_source_num_1 = 0
cnt_source_num_2 = 0
cnt_source_num_3 = 0
cnt_source_num_4 = 0
for file_path in file_path_lst:
    continue_flag = 0
    num_of_sources = file_path.split('-')[-1]
    # load metadata from .json
    with open(json_folder + os.sep + file_path + os.sep + 'sample_log.json') as json_file:
        metadata = json.load(json_file)
    all_sources = []
    # doa2wavid = {}
    doas = []
    wavids = []
    for key in metadata.keys():
        if "source" in key:
            print(f'Path of the component source {metadata[key]["wave_path"]}', flush=True)
            sig = audioread(metadata[key]["wave_path"], 10) # sig.shape: [length, C] ndarray(int16)

            # waveform = activelev(sig.transpose(1, 0))
            waveform = sig.transpose(1, 0) # waveform.shape: (4, 480000)
            # print(f'waveform.shape {waveform.shape}', flush=True)
            # print(f'waveform.dtype {waveform.dtype}', flush=True)
            all_sources.append(waveform)
            doas.append(metadata[key]["angle"])
            wavids.append(metadata[key]["wave_path"].split('/')[-1].split('-')[0])
    all_sources = np.array(all_sources)
    # print(f'all_sources.shape {all_sources.shape}', flush=True) # (4, 4, 480000)
    # print(f'all_sources.dtype {all_sources.dtype}', flush=True) # int16
    channels, frames = all_sources[0].shape

    mixture = np.sum(all_sources, axis=0, dtype=np.int16)
    
    mixture = mixture.transpose(1, 0)
    mixture = np.reshape(mixture, [frames * channels, 1])

    # print(f'mixture.shape {mixture.shape}', flush=True) # (1920000, 1)
    # print(f'mixture.dtype {mixture.dtype}', flush=True) # int16

    print(f'doas {doas}', flush=True)
    print(f'wavids {wavids}', flush=True)
    file_name = '+'.join([str(doa) for doa in doas]) + '-' + '+'.join([wavid for wavid in wavids])

    nos = len(doas)
    output_file_path = output_folder + os.sep + f'SourceNum-{num_of_sources}' + os.sep + file_name + '.wav'
    with wave.open(output_file_path, 'wb') as f:
        f.setframerate(48000)
        f.setsampwidth(2)
        f.setnchannels(channels)
        f.writeframes(mixture.tostring())
    print(f"{output_file_path} has been created!", flush=True)
    
    gts = []
    for (index, wavid) in enumerate(wavids):
        gt_frame_path = single_source_gt_folder + os.sep + wavid.split('_')[0] + os.sep + wavid + '.txt'
        print(f'gt_frame_path {gt_frame_path}', flush=True)
        if not os.path.exists(gt_frame_path):
            continue_flag = 1
            break
        gt = []
        with open(gt_frame_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                gt.append(int(line.split(' ')[1][0]))
        gt = np.array(gt)
        gts.append(gt)
    if continue_flag:
        continue
    gts = np.array(gts) # gts.shape: (nos, num_segs)

    # load signal
    fs, sig = apkit.load_wav(output_file_path)  # sig.shape: [C, length] ndarray(float64)


    feat_seg_idx = 5
    while feat_seg_idx * SEG_HOP + SEG_LEN < sig.shape[1] // 2:
        # computer STFT for each frame
        stft_seg_level = tools.extract_gcc_phat_fb(sig, fs, feat_seg_idx)
        # stft_seg_level.shape: (C*2, num_frames, 337) stft_seg_level.dtype: float64
        
        # Label each frame
        label_seg_level = []
        for (index, doa) in enumerate(doas):
            if gts[index][feat_seg_idx] == 1:
                label_seg_level.append(doa)
        num_sources = len(label_seg_level)
        if num_sources == 0:
            cnt_source_num_0 += 1
        if num_sources == 1:
            cnt_source_num_1 += 1
        if num_sources == 2:
            cnt_source_num_2 += 1
        if num_sources == 3:
            cnt_source_num_3 += 1
        if num_sources == 4:
            cnt_source_num_4 += 1
        # sample_data 同时有特征和标签
        sample_data = {"stft_seg_level" : stft_seg_level, "label_seg_level" : label_seg_level, "num_sources" : num_sources}
        save_path = os.path.join(data_frame_path, '{}_seg_{}.pkl'.format(file_name, feat_seg_idx))
        print(save_path, flush=True)
        # print("sample_data's feat.shape {}".format(sample_data["stft_seg_level"].shape), flush=True) # (6, 40, 51)
        # print("sample_data's label {}".format(sample_data["label_seg_level"]), flush=True)
        # print("sample_data's num_sources {}".format(sample_data["num_sources"]), flush=True)
        pkl_file = open(save_path, 'wb')
        pickle.dump(sample_data, pkl_file)
        pkl_file.close()
        feat_seg_idx += 1
        cnt_segs += 1
    print(f"cnt_source_num_0: {cnt_source_num_0}", flush=True)
    print(f"cnt_source_num_1: {cnt_source_num_1}", flush=True)
    print(f"cnt_source_num_2: {cnt_source_num_2}", flush=True)
    print(f"cnt_source_num_3: {cnt_source_num_3}", flush=True)
    print(f"cnt_source_num_4: {cnt_source_num_4}", flush=True)
    print(f"cnt_segs: {cnt_segs}", flush=True)