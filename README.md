# VCTK-SIM

## Usage

### 1. Download the simulated dataset

Download link: https://drive.google.com/drive/folders/1CXcJCXGSy76LoO_nX8C7w84fZpirkH2d?usp=sharing



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
│   ├── mulch_data_2rooms_test5 (dev set)
```

VCTK_3mix and VCTK_4mix include samples containing 0,1,2,3 and 0,1,2,3,4, respectively in json format.

```
├── VCTK_3/4mix
│   ├── json_3/4sources_20000_50roomsA (train set)
│   ├── json_3/4sources_2000_test_6 (test set)
│   ├── json_3/4sources_2000_test_5 (dev set)
```

Scripts

sim_mix_audio.py
