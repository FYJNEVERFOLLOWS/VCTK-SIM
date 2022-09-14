import torch
from speechbrain.pretrained import VAD
import os
import glob


VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")

files = glob.glob('/Work18/2021/fuyanjie/datasets/VCTK-Corpus/wav16/*/*.wav')

output_folder = '/Work18/2021/fuyanjie/exp_data/vad_vctk/'

for audio_file in files:
    # 1- Let's compute frame-level posteriors first
    # audio_file = '/content/drive/My Drive/pretrained_models/speechbrain_workspace/vad-crdnn-libriparty/16k_p241_002_mic2.flac'
    try:
        prob_chunks = VAD.get_speech_prob_file(audio_file)

        # 2- Let's apply a threshold on top of the posteriors
        prob_th = VAD.apply_threshold(prob_chunks).float()

        # 3- Let's now derive the candidate speech segments
        boundaries = VAD.get_boundaries(prob_th)

        # 4- Apply energy VAD within each candidate speech segment (optional)
        # boundaries = VAD.energy_VAD(audio_file,boundaries)

        # 5- Merge segments that are too close
        # boundaries = VAD.merge_close_segments(boundaries, close_th=0.250)

        # 6- Remove segments that are too short
        # boundaries = VAD.remove_short_segments(boundaries, len_th=0.250)

        # 7- Double-check speech segments (optional).
        # boundaries = VAD.double_check_speech_segments(boundaries, audio_file,  speech_th=0.5)
        
        file_name = str(audio_file).split('/')[-1]
        spk_id = file_name.split('_')[0]
        save_folder = output_folder + spk_id + '/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = save_folder + file_name.replace('.wav', '.txt')
        print('save_path', save_path, flush=True)
        # Print the output
        VAD.save_boundaries(boundaries, save_path=save_path)
    except Exception as e:
        continue