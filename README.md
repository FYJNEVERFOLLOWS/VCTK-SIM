# VCTK-SIM

## Citation
Please cite the following paper if you use our dataset:
1. Fu, Y., Ge, M., Yin, H., Qian, X., Wang, L., Zhang, G., Dang, J. (2022) Iterative Sound Source Localization for Unknown Number of Sources. Proc. Interspeech 2022, 896-900, doi: 10.21437/Interspeech.2022-10525
```bibtex
@inproceedings{fu22c_interspeech,
  author={Yanjie Fu and Meng Ge and Haoran Yin and Xinyuan Qian and Longbiao Wang and Gaoyan Zhang and Jianwu Dang},
  title={{Iterative Sound Source Localization for Unknown Number of Sources}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={896--900},
  doi={10.21437/Interspeech.2022-10525}
}
```

## Usage

### 1. Download the simulated dataset

Download link: https://pan.baidu.com/s/1N6xWJfLUipIfZTi7QMu_jQ?pwd=ohaq 


Structures of the directories:

```
├── ISSL_dataset
│   ├── SIM_Rooms.zip
│   ├── VCTK_3mix.zip
│   └── VCTK_4mix.zip
```

SIM_Rooms includes three directories below:

```
├── SIM_Rooms
│   ├── mulch_data_50rooms_trainA (train set)
│   ├── mulch_data_2rooms_test6 (test set)
│   └── mulch_data_2rooms_test5 (dev set)
```

VCTK_3mix and VCTK_4mix include samples containing 0,1,2,3 and 0,1,2,3,4, respectively, in json format.

```
├── VCTK_3/4mix
│   ├── json_3/4sources_20000_50roomsA (train set)
│   ├── json_3/4sources_2000_test_6 (test set)
│   └── json_3/4sources_2000_test_5 (dev set)
```

### 2. VAD for VCTK at segment-level (8192 samples)
Directly download `vctk_gt_frame` folder or run the scripts below:
```bash
pip install speechbrain
cd speechbrain_vad
python prep_for_vctk.py # convert 48kHz VCTK to 16kHz
python sb_vad.py # vad using speechbrain
python label_vctk_at_seg_level.py # generate speech-nonspeech label for VCTK at segment-level
```


### 3. Mix audio and slice into segment-wise samples

run `sim_mix_audio.py` for different json folders and get  `sim_3/4sources_test_5/6_data` and  `sim_50rooms_2W_A_data`, respectively.

All samples are stored in `.pkl` format.
