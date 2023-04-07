import argparse
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import time
from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
import sys

sys.path.append("/Work18/2021/fuyanjie/pycode/SimMulChanData")

import numpy as np
import soundfile as sf

from gen_room_para import gen_room_para, gen_mulchannel_data_random, gen_mulchannel_data_angle
from save_mul_ch_audio import trans_save_mul_ch_audio
from gen_file_lst import generate_wav_list_from_lst_file
from tools import get_first_level_folder

MAX_WORKERS = max(mp.cpu_count() - 5, 10)


def __sim_stationary_noise(folder, max_gen_num, wav_list, fs, speech_length):
    room_para_path = os.path.join(folder, 'room_para.npy')
    room_para = np.load(room_para_path, allow_pickle=True).item()
    print(f'room_para {room_para}', flush=True)
    used_wav_num = random.randint(1, 3)
    wav_samples = random.sample(wav_list, used_wav_num)
    print(f'num of stationary_noise wav_samples {len(wav_samples)}', flush=True)

    # judge if multi-channel stationary noise wav files exist
    # file_list = os.listdir(os.path.join(folder, 'stationary_noise'))
    st_noise_dir = os.path.join(folder, 'stationary_noise')
    if os.path.exists(st_noise_dir) and len(os.listdir(st_noise_dir)) > 0:
        print(f'{st_noise_dir} is not empty!', flush=True)
        return

    for num in range(max_gen_num):
        # 生成多个多通道平稳噪声数据，这些噪声的 source_location 不同
        mulchannel_audio_data_list = []
        for wav_file in wav_samples:
            multichannel_audio_data, _, angle_degree = gen_mulchannel_data_random(wav_file.strip(),
                                                                                room_para,
                                                                                folder,
                                                                                audio_type=0,
                                                                                fs=fs,
                                                                                segment_length=speech_length)
            multichannel_audio_data = multichannel_audio_data[:, 0:fs * speech_length]
            mulchannel_audio_data_list.append(multichannel_audio_data)

        # 对生成数据幅度进行一下调整，再进行叠加
        out_mulchannel_audio_data = np.zeros_like(mulchannel_audio_data_list[0], dtype=np.int16)
        for data in mulchannel_audio_data_list:
            out_mulchannel_audio_data += data

        # 将生成好的数据进行保存
        mic_num, _ = out_mulchannel_audio_data.shape
        save_folder = os.path.join(folder, 'stationary_noise')
        output_file_path = os.path.join(save_folder, 'stationary_noise_SourceNum-{}_{}.wav'.format(used_wav_num, num))
        trans_save_mul_ch_audio(out_mulchannel_audio_data, output_file_path, fs=fs)


def sim_stationary_noise(base_dir, stationary_noise_lst_file, sim_room_num, max_stationary_noise_num, fs, room_transpose_prob, speech_length):
    '''
    利用多个平稳噪声模拟真实环境下的多通道平稳噪声，每个房间就仿真一条数据，主要目的是为了更好模拟不同环境下的平稳噪声
    :param base_dir: 根目录，用于保存不同房间的数据
    :param stationary_noise_lst_file: 平稳噪声列表文件
    :param sim_room_num: 模拟不同房间的数目
    :param max_stationary_noise_num: 最多使用平稳噪声的数目
    :param fs:
    :return:
    '''
    for i in range(sim_room_num):
        _ = gen_room_para(base_dir, room_transpose_prob=room_transpose_prob)

    room_folders = get_first_level_folder(base_dir)
    wav_list = list(generate_wav_list_from_lst_file(stationary_noise_lst_file))
    print(f'wav_list {wav_list}', flush=True)

    # without using multi-processing
    # for folder in room_folders:
    #     __sim_stationary_noise(folder, max_gen_num=max_stationary_noise_num, wav_list=wav_list,
    #                    fs=fs, speech_length=speech_length)
    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        func = partial(__sim_stationary_noise, max_gen_num=max_stationary_noise_num, wav_list=wav_list,
                       fs=fs, speech_length=speech_length)
        ex.map(func, room_folders)


def __sim_speech(folder, wav_list, fs, cover_angle_range, multi_list, segment_length, speaker_list, angular_spacing):
    room_para_path = os.path.join(folder, 'room_para.npy')
    room_para = np.load(room_para_path, allow_pickle=True).item()

    # 声明多通道人声数据保存路径
    mulchannel_speech_folder = os.path.join(folder, 'speech')

    cover_angle_num_float = (cover_angle_range[1] - cover_angle_range[0] + 1) / angular_spacing
    cover_angle_num = math.floor(cover_angle_num_float)


    speaker_folder = ""
    speaker_folders = []
    speaker_lsts = []
    seen_speaker_lsts = []

    with open(speaker_list) as fid:
        for line in fid:
            seen_speaker_lsts.append(line.strip())

    if not multi_list:
        speaker_folder = os.path.dirname(wav_list[0])
        speaker_folder = os.path.dirname(speaker_folder)
        speaker_lst = os.listdir(speaker_folder)
        print(f'speaker_folder {speaker_folder}', flush=True)
        speaker_idx_lst = np.arange(len(speaker_lst))
        print(f'len(speaker_lst) {len(speaker_lst)}', flush=True)
        np.random.shuffle(speaker_idx_lst)
        # speaker_idx_lst = speaker_idx_lst[:cover_angle_num]
        speaker_idx_lst = speaker_idx_lst[:]
        print(f'len(speaker_idx_lst) {len(speaker_idx_lst)}', flush=True)
    else:
        for wav_lst in wav_list:
            speaker_folder = os.path.dirname(wav_lst[0])
            speaker_folder = os.path.dirname(speaker_folder)
            speaker_lst = os.listdir(speaker_folder)
            speaker_folders.append(speaker_folder)
            speaker_lsts = list(set(speaker_lsts + speaker_lst))
        speaker_lst = list(set(speaker_lsts).intersection(set(seen_speaker_lsts)))
        speaker_idx_lst = np.arange(len(speaker_lst))
        np.random.shuffle(speaker_idx_lst)
        speaker_idx_lst = speaker_idx_lst[:cover_angle_num]

    # get the wav_id list
    wav_id_file_path = os.path.join(folder, 'speech_wav_id_lst.lst')
    if os.path.exists(wav_id_file_path):
        wav_id_lst = []
        with open(wav_id_file_path, 'r', encoding='utf8') as f:
            wav_id_set = f.readlines()

            for wav_id in wav_id_set:
                wav_id = wav_id.strip()
                wav_id_lst.append(wav_id)
    else:
        wav_id_lst = []
    # print(f'wav_id_lst {wav_id_lst}', flush=True)
    angle = cover_angle_range[0]
    random_angle = cover_angle_range[0]
    index = 0
    # print(f'speaker_idx_lst {speaker_idx_lst}', flush=True)

    while random_angle < 360:
        k = speaker_idx_lst[index % len(speaker_idx_lst)]

        if len(speaker_folders) > 1:
            if random.uniform(0, 1) > 0:
                exist_lst = []
                for speaker_folder_tmp in speaker_folders:
                    if os.path.exists(os.path.join(speaker_folder_tmp, speaker_lst[k])):
                        exist_lst.append(speaker_folder_tmp)
                speaker_folder = random.sample(exist_lst, 1)
                speaker_folder = speaker_folder[0]

        path = os.path.join(speaker_folder, speaker_lst[k])

        wav_lst = os.listdir(os.path.join(speaker_folder, speaker_lst[k]))

        print(f'path {path}', flush=True)
        # print(f'wav_lst {wav_lst}', flush=True)

        file_name = random.sample(wav_lst, 1)
        print(f'file_name {file_name}', flush=True)

        print(f'wav_id_lst {wav_id_lst}', flush=True)
        if speaker_lst[k] not in wav_id_lst:
            wav_path = os.path.join(speaker_folder, speaker_lst[k])
            wav_path = os.path.join(wav_path, file_name[0])

            random_range = random.randint(0, angular_spacing - 1)
            random_angle = angle + random_range

            if speaker_lst[k] not in wav_id_lst:
                wav_id_lst.append(speaker_lst[k])
            for dis_idx in range(3):


                multichannel_audio_data, wav_id, distance = gen_mulchannel_data_angle(wav_path,
                                                                                      room_para,
                                                                                      folder,
                                                                                      angle=random_angle,
                                                                                      distance_flag=dis_idx,
                                                                                      fs=fs,
                                                                                      segment_length=segment_length,
                                                                                      audio_type=1)

                max_speech_value = np.max(np.abs(multichannel_audio_data))

                if max_speech_value > 32767:
                    multichannel_audio_data = multichannel_audio_data / max_speech_value * 30000
                out_mulchannel_audio_data = multichannel_audio_data

                # 生成文件保存路径
                if dis_idx == 0:
                    distance_type = "close"
                elif dis_idx == 1:
                    distance_type = "middle"
                else:
                    distance_type = "far"
                file_name = "{}-{}m_{}.wav".format(wav_id, distance, distance_type)
                angle_speech_folder = os.path.join(mulchannel_speech_folder, str(random_angle % 360))
                save_file_path = os.path.join(angle_speech_folder, file_name)

                trans_save_mul_ch_audio(out_mulchannel_audio_data, save_file_path, fs=fs)
                # print(save_file_path + " finished!", flush=True)

            angle = random_angle + angular_spacing
            index = index + 1
            print(f'random_angle {random_angle}', flush=True)
        else:
            continue
    print(f"wav_id_lst created! {len(wav_id_lst)}", flush=True)
    # wait for all the wav simulation finish, write wav_id_lst in the wav_id_file_path
    with open(wav_id_file_path, 'w', encoding='utf8') as f:
        for wav_id in wav_id_lst:
            f.write(wav_id + '\n')

def sim_speech(base_dir, mono_speech_lst_file, fs, cover_angle_range, segment_length, speaker_list,
               angular_spacing):
    '''
    模拟多通道人声数据
    :param base_dir: 根目录
    :param mono_speech_lst_file: 单通道人声文件列表文件
    :param max_wav_num:
    :param fs:
    :return:
    '''
    room_folders = get_first_level_folder(base_dir)
    wav_list = []

    if len(mono_speech_lst_file) == 1:
        print(f'mono_speech_lst_file[0] {mono_speech_lst_file[0]}', flush=True)
        wav_list = list(generate_wav_list_from_lst_file(mono_speech_lst_file[0]))
        
        # without using multi-processing
        # for folder in room_folders:
        #     __sim_speech(folder, wav_list=wav_list, fs=fs, cover_angle_range=cover_angle_range, multi_list=False,
        #                    segment_length=segment_length, speaker_list=speaker_list, angular_spacing=angular_spacing)

        with ProcessPoolExecutor(MAX_WORKERS) as ex:
            func = partial(__sim_speech, wav_list=wav_list, fs=fs, cover_angle_range=cover_angle_range, multi_list=False,
                           segment_length=segment_length, speaker_list=speaker_list, angular_spacing=angular_spacing)
            ex.map(func, room_folders)


def __sim_non_stationary_noise(folder, sim_wav_num, wav_list, fs, cover_angle_range, speech_length, angular_spacing):
    room_para_path = os.path.join(folder, 'room_para.npy')
    room_para = np.load(room_para_path, allow_pickle=True).item()

    speech_path = os.path.join(folder, 'speech')
    angle_of_speech = os.listdir(speech_path)
    angle_of_speech = list(map(int, angle_of_speech))
    angle_of_speech.sort()
    print(f'angle_list_of_speech: {angle_of_speech}', flush=True)

    # 声明多通道非平稳噪声数据保存路径
    mulchannel_nonstationary_noise_folder = os.path.join(folder, 'nonstationary_noise')

    cover_angle_num = (cover_angle_range[1] - cover_angle_range[0] + 1) / angular_spacing
    cover_angle_num = math.floor(cover_angle_num)

    non_stationary_list = wav_list
    np.random.shuffle(non_stationary_list)
    non_stationary_list = non_stationary_list[:3 * cover_angle_num]

    angle = angle_of_speech[0]
    index = 0
    for wav in non_stationary_list:
        multichannel_audio_data, wav_id, distance = gen_mulchannel_data_angle(wav.strip(),
                                                                              room_para,
                                                                              folder,
                                                                              angle=angle,
                                                                              distance_flag=index % 3,
                                                                              fs=fs,
                                                                              segment_length=speech_length,
                                                                              audio_type=2)

        # out_mulchannel_audio_data = multichannel_audio_data - np.mean(multichannel_audio_data, axis=0)
        out_mulchannel_audio_data = multichannel_audio_data

        # 生成文件保存路径
        if index % 3 == 0:
            distance_type = "close"
        elif index % 3 == 1:
            distance_type = "middle"
        else:
            distance_type = "far"
        file_name = "{}-{}m_{}.wav".format(wav_id, distance, distance_type)
        angle_speech_folder = os.path.join(mulchannel_nonstationary_noise_folder, str(angle))

        save_file_path = os.path.join(angle_speech_folder, file_name)

        trans_save_mul_ch_audio(out_mulchannel_audio_data, save_file_path, fs=fs)
        # print(save_file_path + " finish!", flush=True)

        if index % 3 == 2:
            angle = angle + angular_spacing
        index = index + 1



def sim_non_stationary_noise(base_dir, mono_nonstationary_noise_lst_file, max_wav_num, fs, cover_angle_range,
                             speech_length, angular_spacing):
    '''
    模拟多通道非平稳噪声
    :param base_dir: 根目录
    :param mono_nonstationary_noise_lst_file: 单通道非平稳噪声文件列表文件
    :param max_wav_num:
    :param fs: 采样率
    :return:
    '''
    room_folders = get_first_level_folder(base_dir)

    wav_list = list(generate_wav_list_from_lst_file(mono_nonstationary_noise_lst_file))
    if len(wav_list) > max_wav_num:
        sim_wav_num = max_wav_num
    else:
        sim_wav_num = len(wav_list)

    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        func = partial(__sim_non_stationary_noise, sim_wav_num=sim_wav_num, wav_list=wav_list, fs=fs,
                       cover_angle_range=cover_angle_range, speech_length=speech_length,
                       angular_spacing=angular_spacing)
        ex.map(func, room_folders)


def single_proc(stationary_noise_lst, speech_lst, non_stationary_noise_lst, save_path,
                sim_room_num, stationary_noise_num, fs, speech_length, cover_angle_range, room_transpose_prob, speaker_list, augular_spacing):
    print("MAX_WORKERS:", MAX_WORKERS, flush=True)

    # Start timing¬
    start_time = time.time()

    # make directory of output data
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print(f'{save_path} already exists!', flush=True)
        shutil.rmtree(save_path)
        print(f'remove {save_path}', flush=True)

    # generate room parameters and stationary noise
    sim_stationary_noise(save_path, stationary_noise_lst, sim_room_num, stationary_noise_num, fs, room_transpose_prob, speech_length)
    print('Finish sim_stationary_noise', flush=True)

    # generate multi-channel speech
    sim_speech(save_path, speech_lst, fs, cover_angle_range, speech_length, speaker_list, augular_spacing)
    print('Finish sim_speech', flush=True)

    # DO NOT INVOKE FOR NOW!
    # generate multi-channel non-stationary noise
    # sim_non_stationary_noise(save_path, non_stationary_noise_lst, non_stationary_num, fs, cover_angle_range, speech_length, augular_spacing)
    # print('Finish sim_non_stationary_noise', flush=True)

    # End timing
    print('time spent: %s' % str(time.time() - start_time), flush=True)


if __name__ == '__main__':
    '''
    convert single channel format to multi-channel(6-channel) format of single voice.

    :param stationary_noise_lst: contains the absolute path of stationary noise .wav
    :param speech_lst: contains the absolute path of clean speech .wav
    :param non_stationary_noise_lst: contains the absolute path of non-stationary noise .wav
    :param save_path: the output path of multi-channel data
    :param sim_room_num: the number of simulated rooms, each room contains specified number of stationary noise, speech, and non-stationary noise
    :param stationary_noise_num: the upper bound of the classes of stationary noise we choose
    :param speech_num: the number of simulated speech in each room
    :param non_stationary_num: the number of simulated non-stationary noise in each room
    :param fs: sample rate
    :return: 
    '''

    # 设置路径参数
    stationary_noise_lst = '/Work18/2021/fuyanjie/exp_data/sim_audio_vctk/lst/noise_stationary_lst.lst'
    speech_lsts = ['/Work18/2021/fuyanjie/exp_data/sim_audio_vctk/lst/clean_speech_lst.lst']
    non_stationary_noise_lst = '/Work18/2021/fuyanjie/exp_data/sim_audio_vctk/lst/noise_non_stationary_lst.lst'

    seen_speaker_lst = '/Work18/2021/fuyanjie/exp_data/sim_audio_vctk/lst/seen_speaker.lst' # for training
    # seen_speaker_lst = '/Work18/2021/fuyanjie/exp_data/sim_audio_vctk/lst/seen_speaker_all.lst' # for training
    unseen_speaker_lst = '/Work18/2021/fuyanjie/exp_data/sim_audio_vctk/lst/unseen_speaker.lst' # for testing
    
    # save_path = '/Work18/2021/fuyanjie/exp_data/sim_audio_vctk/mulch_data_2rooms'

    # 设定合成参数
    # sim_room_num = 2
    # stationary_noise_num = 5
    # fs = 48000
    # speech_length = 10
    cover_angle_range = [0, 360]
    # angular_spacing = 5 # 角度最小间隔
    # room_transpose_prob = 0.5

    parser = argparse.ArgumentParser(description='Simulate multi-channel audio')
    parser.add_argument('--save_path', metavar='SAVE_PATH', type=str,
                        help='path to the save the simulated audio')
    parser.add_argument('--speaker_list', metavar='SPEAKER_LIST', type=str,
                        default=unseen_speaker_lst, help='path to the speaker list')
    parser.add_argument('--sim_room_num', metavar='SIM_ROOM_NUM', type=int,
                        default=2, help='number of rooms to simulate')
    parser.add_argument('--stationary_noise_num', metavar='STATIONARY_NOISE_NUM', type=int,
                        default=5, help='the upper bound of the classes of stationary noise we choose')
    parser.add_argument('--fs', metavar='FS', type=int,
                        default=48000, help='sample rate')
    parser.add_argument('--speech_length', metavar='SPEECH_LENGTH', type=int,
                        default=10, help='the length of speech in seconds')                    
    parser.add_argument('--angular_spacing', metavar='ANGULAR_SPACING', type=int,
                        default=5, help='minimum angular spacing between every two sources')
    parser.add_argument('--room_transpose_prob', metavar='ROOM_TRANSPOSE_PROB', type=int,
                        default=0.5, help='the probability to switch the length and width')

    args = parser.parse_args()

    # Print arguments
    print('========== Print arguments ==========', flush=True)
    for k, v in vars(args).items():
        print(k,' = ',v, flush=True)
    print('========== Print arguments ==========', flush=True)
    single_proc(stationary_noise_lst, speech_lsts, non_stationary_noise_lst, args.save_path, args.sim_room_num,
                args.stationary_noise_num, args.fs, args.speech_length, cover_angle_range, args.room_transpose_prob,
                args.speaker_list, args.angular_spacing)

