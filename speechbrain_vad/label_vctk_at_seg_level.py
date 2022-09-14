import os
from pathlib import Path
from collections import deque

SEG_LEN = 8192
SEG_HOP = SEG_LEN // 2

files = list(Path('/Work18/2021/fuyanjie/exp_data/vad_vctk/').rglob('*.txt'))
save_dir = f'/Work18/2021/fuyanjie/exp_data/vctk_gt_frame_{SEG_LEN}/'

for f in files:
    gt_file = str(f).replace('vad_vctk', f'vctk_gt_frame_{SEG_LEN}')

    file_name = str(f).split('/')[-1]
    sub_folder = file_name.split('_')[0]
    # sub_folder is namely the spk_id
    dst_folder = os.path.join(save_dir, sub_folder)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    file_id = file_name.split('.')[0]

    # 分别存放 speech_seg 和 nonspeech_seg 开始的时间
    speech_queue = deque()
    nonspeech_queue = deque()
    with open(str(f), "r") as f:
        lines = f.readlines()
        print(f'len(lines) {len(lines)}', flush=True)
        if len(lines) == 0:
            continue
        for row, line in enumerate(lines):
            seg_data = line.split(' ')
            print(f'seg_data {seg_data}', flush=True)

            if seg_data[-1].startswith('S'):
                speech_queue.append(float(seg_data[2]) * 48000)
            elif seg_data[-1].startswith('N'):
                nonspeech_queue.append(float(seg_data[2]) * 48000)
            if row == len(lines) - 1:
                nonspeech_queue.append(float(seg_data[4]) * 48000)
    print(f'speech_queue {speech_queue}', flush=True)
    print(f'nonspeech_queue {nonspeech_queue}', flush=True)

    with open(gt_file, 'w') as fp:
        # flag for speech or nonspeech
        flag_son = 0
        # 10s * 48kHz
        for idx_sample in range(0, 480000 - SEG_LEN, SEG_HOP):
            idx_mid = idx_sample + SEG_HOP
            idx_frame = idx_sample // SEG_HOP
            if speech_queue or nonspeech_queue:
                if nonspeech_queue and idx_mid > nonspeech_queue[0]:
                    if flag_son == 1 and speech_queue:
                        speech_queue.popleft()
                    flag_son = 0
                if speech_queue and idx_mid > speech_queue[0]:
                    if flag_son == 0 and nonspeech_queue:
                        nonspeech_queue.popleft()
                    flag_son = 1

                # print(f'idx_frame {idx_frame} speech_queue {speech_queue} nonspeech_queue {nonspeech_queue}', flush=True)
                    
                fp.write(f'{idx_frame} {flag_son}\n')

    print(f'Successfully saved to gt_file: {gt_file}', flush=True)
