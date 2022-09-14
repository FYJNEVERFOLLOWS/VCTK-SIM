# -*- coding:utf-8 -*-
import os
import glob
from pathlib import Path
import shutil
import librosa
import soundfile as sf
import numpy as np

# 两个参数：你要遍历的目录，匹配的正则表达式
# files = list(Path('/Work18/2021/fuyanjie/datasets/VCTK-Corpus/wav48').rglob('*.wav'))
# files = list(Path('/CDShare3/VCTK-Corpus-backup/wav48').rglob('*/*.wav'))
files = glob.glob('/Work18/2021/fuyanjie/datasets/VCTK-Corpus/wav48/*/*.wav')
save_dir = '/Work18/2021/fuyanjie/datasets/VCTK-Corpus/wav16'
print('len of files {}'.format(len(files)))

for f in files:
    file_name = str(f).split('/')[-1]
    sub_folder = file_name.split('_')[0]
    # sub_folder is namely the spk_id
    dst_folder = os.path.join(save_dir, sub_folder)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    dst = os.path.join(dst_folder, file_name)

    sig, fs = sf.read(f, dtype='float32') # 一定要对数据做astype(np.float32)，否则会出现下采样无效。
    sig_16k = librosa.resample(sig.astype(np.float32)*32767, fs, 16000) # librosa自动做了归一化，需要对librosa读取到的数据data*32767.
    sig_16k = sig_16k.astype(np.int16)
    sf.write(dst, sig_16k, 16000)
    print('Successfully saved to {}!'.format(dst))


