import os
import re
import wave
import numpy as np
import apkit

_FREQ_MAX = 8000
_FREQ_MIN = 100
SEG_LEN = 8192
SEG_HOP = 4096

def get_first_level_folder(dir):
    folder_list = []
    for entry in os.scandir(dir):
        if entry.is_dir():
            folder_list.append(entry.path)

    return folder_list

def get_first_level_order_with_rg(path):
    dirs = os.listdir(path)
    print(type(dirs))
    for dir in dirs:
        res = re.search("^[0-9]{8}$", dir)
        if res:
            res = res.group()
        else:
            continue
        print(res)

def raw2wav(inf_path, outf_path,sample_rate = 16000):

    pcmfile = open(inf_path, 'rb')
    pcmdata = pcmfile.read()
    wavfile = wave.open(outf_path, 'wb')
    wavfile.setframerate(sample_rate)
    wavfile.setsampwidth(2)    #16位采样即为2字节
    wavfile.setnchannels(1)
    wavfile.writeframes(pcmdata)
    wavfile.close()

def extract_stft_real_image(sig, fs, feat_seg_idx):
    # calculate the complex spectrogram stft
    tf = apkit.stft(sig[:, feat_seg_idx * SEG_HOP : feat_seg_idx * SEG_HOP + SEG_LEN], apkit.cola_hamming, 2048, 1024, last_sample=True)
    # tf.shape: [C, num_frames, win_size] tf.dtype: complex128
    nch, nframe, _ = tf.shape # num_frames=sig_len/hop_len - 1
    # tf.shape:(4, num_frames, 2048) num_frames=7 when len_segment=8192 and win_size=2048
    # why not Nyquist 1 + n_fft/ 2?

    # trim freq bins
    max_fbin = int(_FREQ_MAX * 2048 / fs)            # 100-8kHz
    min_fbin = int(_FREQ_MIN * 2048 / fs)            # 100-8kHz
    tf = tf[:, :, min_fbin:max_fbin]
    # tf.shape: (C, num_frames, 337)

    # calculate the magnitude of the spectrogram
    mag_spectrogram = np.abs(tf)
    # print(f'mag_spectrogram.shape {mag_spectrogram.shape} mag_spectrogram.dtype {mag_spectrogram.dtype}')
    # mag_spectrogram.shape: (C, num_frames, 337) mag_spectrogram.dtype: float64

    # calculate the phase of the spectrogram
    phase_spectrogram = np.angle(tf)
    # print(f'phase_spectrogram.shape {phase_spectrogram.shape} phase_spectrogram.dtype {phase_spectrogram.dtype}')
    # imaginary_spectrogram.shape: (C, num_frames, 337) imaginary_spectrogram.dtype: float64

    # combine these two parts by the channel axis
    stft_seg_level = np.concatenate((mag_spectrogram, phase_spectrogram), axis=0)
    # print(f'stft_seg_level.shape {stft_seg_level.shape} stft_seg_level.dtype {stft_seg_level.dtype}')
    # stft_seg_level.shape: (C*2, num_frames, 337) stft_seg_level.dtype: float64

    return stft_seg_level

def extract_gcc_phat_fb(sig, fs, feat_seg_idx):
    # calculate the complex spectrogram stft
    tf = apkit.stft(sig[:, feat_seg_idx * SEG_HOP : feat_seg_idx * SEG_HOP + SEG_LEN], apkit.cola_hamming, 2048, 1024, last_sample=True)
    # tf.shape: [C, num_frames, win_size] tf.dtype: complex128
    nch, nframe, _ = tf.shape # num_frames=sig_len/hop_len - 1
    # tf.shape:(4, num_frames, 2048) num_frames=7 when len_segment=8192 and win_size=2048

    # trim freq bins
    max_fbin = int(_FREQ_MAX * 2048 / fs)            # 100-8kHz
    min_fbin = int(_FREQ_MIN * 2048 / fs)            # 100-8kHz
    freq = np.fft.fftfreq(2048)[min_fbin:max_fbin]
    tf = tf[:, :, min_fbin:max_fbin]

    # compute pairwise gcc on f-banks
    ecov = apkit.empirical_cov_mat(tf, fw=1, tw=1)
    nfbank = 40
    zoom = 25
    eps = 0.0

    fbw = apkit.mel_freq_fbank_weight(nfbank, freq, fs, fmax=_FREQ_MAX,
                                        fmin=_FREQ_MIN)
    fbcc = apkit.gcc_phat_fbanks(ecov, fbw, zoom, freq, eps=eps)

    # merge to a single numpy array, indexed by 'tpbd'
    #                                           (time=num_frames, pair=6, bank=40, delay=51)
    feature = np.asarray([fbcc[(i,j)] for i in range(nch)
                                        for j in range(nch)
                                        if i < j])
    feature = np.moveaxis(feature, 2, 0)

    # and map [-1.0, 1.0] to 16-bit integer, to save storage space
    dtype = np.int16
    vmax = np.iinfo(dtype).max
    feature = (feature * vmax).astype(dtype) # feature.shape: (num_frames, 6, 40, 51)
    gcc_fbank_seg_level = feature[0] # gcc_fbank_seg_level.shape: (6, 40, 51)

    return gcc_fbank_seg_level


if __name__ == '__main__':
    # print(get_first_level_folder('/Users/fuyanjie/Desktop/PG/Audio&Speech'))
    # print(get_first_level_folder('/Users/fuyanjie/Desktop/PG/Audio&Speech').__len__())
    print(get_first_level_order_with_rg("/Users/fuyanjie/Desktop/temp/audio_hw/speech/2s"))

    # indir = "/Users/fuyanjie/Desktop/temp/StationaryNoise/N101-N115 noises(raw, 16k, 16bit, mono)"
    # outdir = "/Users/fuyanjie/Desktop/temp/StationaryNoise/N101-N115 noises(wav, 16k, 16bit, mono)"
    # files = os.listdir(indir)
    # for file in files:
    #     input = os.path.join(indir, file)
    #     output = os.path.join(outdir, file.replace("raw", "wav"))
    #     raw2wav(input, output, 16000)

