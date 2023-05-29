import wave
import numpy as np


def trans_save_mul_ch_audio(mul_ch_audio_data, output_file_path, fs=16000):
    channels, frames = mul_ch_audio_data.shape
    mul_ch_audio_data = mul_ch_audio_data.transpose(1, 0)
    # print(f'mul_ch_audio_data.shape {mul_ch_audio_data.shape}') # (160000, 6)
    out_data = np.reshape(mul_ch_audio_data, [frames * channels, 1])
    out_data = out_data.astype(np.int16)

    # print(f'out_data.shape {out_data.shape}') # (960000, 1)

    with wave.open(output_file_path, 'wb') as f:
        f.setframerate(fs)
        f.setsampwidth(2)
        f.setnchannels(channels)
        f.writeframes(out_data.tostring())
        f.close()
    print(f"{output_file_path} has been created!", flush=True)

def save_audio_separately(audio, path, fs):
    channels, length = audio.shape

    for channel in range(channels):
        suffix = '_multichannel_' + str(channel) + '.wav'

        out_data = np.reshape(audio[channel, :], [length, 1])

        out_data = out_data.astype(np.int16)

        with wave.open(path+suffix, 'wb') as f:
            f.setframerate(fs)
            f.setsampwidth(2)
            f.setnchannels(1)
            f.writeframes(out_data.tostring())
        # f.close()