# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 13:33:48 2025

@author: Fujie
"""
"""
模块用于从 MATLAB 数据文件中加载 EEG 试验数据，转换为 MNE 的 Epochs 格式，并将其保存为 FIF 文件。
"""

import logging
from pathlib import Path
import openpyxl
import pandas as pd
import mne
import numpy as np
import scipy.io as sio
import h5py

ROOT = Path('./')
SAVE_DIR = ROOT / '1_epoch_data/'
RAW_DATA_DIR = ROOT / "Track#3 Imagined speech classification"
FOLDER_NAME_LIST = ['Training set', 'Validation set', 'Test set']
EVENT_ID = {"Hello": 0, "Help me": 1, "Stop": 2, "Thank you": 3, "Yes": 4}


def hdf5_cell_strings(ds):
    """
    解码 MATLAB cell array of char。

    从 HDF5 数据集中读取 cell 引用，解码每个引用对应的字符数组，返回 Python 字符串列表。

    Args:
        ds (h5py.Dataset): 包含 MATLAB cell array of char 的 HDF5 数据集。

    Returns:
        List[str]: 解码后的字符串列表。
    """
    raw = ds[()]
    strings = []
    for ref in raw.flat:
        arr = ds.file[ref][()]  # 比如 uint16 数组
        # 如果是 uint16，每个元素是一个字符编码：
        try:
            s = ''.join(chr(c) for c in arr.flatten())
        except TypeError:
            # 如果不是数字编码，直接把 bytes decode：
            s = arr.tobytes().decode('utf-16le', errors='ignore')
        strings.append(s)
    return strings


def hdf5_char_array(ds):
    """
    解码 MATLAB 的 char array。

    将 HDF5 数据集中存储的字符数组（数字或 bytes）解码为单个 Python 字符串。

    Args:
        ds (h5py.Dataset): 包含 MATLAB char array 的 HDF5 数据集。

    Returns:
        str: 解码后的字符串。
    """
    arr = ds[()]
    # 数字数组（uint16）：
    if np.issubdtype(arr.dtype, np.integer):
        return ''.join(chr(c) for c in arr.flatten())
    # bytes 数组（变长 string）：
    b = arr.tobytes()
    # MATLAB 存的是 utf-16le：
    return b.decode('utf-16le', errors='ignore').rstrip('\x00')


def load_test(folder_dir, sub_i, event_id, key='epo_test'):
    """
    加载测试集数据并构造 MNE EpochsArray。

    从指定目录加载测试集的 MAT/HDF5 文件，解码通道名、坐标、数据和事件标签，
    并返回符合 MNE 格式的 EpochsArray 对象。

    Args:
        folder_dir (str or Path): 测试数据所在目录。
        sub_i (int): 被试编号（从 1 开始）。
        event_id (dict): 事件标识映射字典。
        key (str): MAT 文件中 epoch 数据的键，默认为 'epo_test'。

    Returns:
        mne.EpochsArray: 加载并转换后的测试集 Epochs 对象。

    Raises:
        IndexError: 如果在 Answer Sheet 中找不到对应的被试列。
    """
    matfile_path = Path(folder_dir) / f"Data_Sample{sub_i:02d}.mat"

    # 1. 打开文件（自动 close）
    with h5py.File(str(matfile_path), 'r') as mat_data:

        # 2. 读取epoch数据
        epo = mat_data[key]
        mnt = mat_data['mnt']

        # 3. 读取通道名（cell array）
        clab = hdf5_cell_strings(mnt['clab'])
        pos3d = mnt['pos_3d'][()]  # (64*3)

        # 4. 读取采样率、标题
        fs = float(epo['fs'][()].item())     # scalar
        title = hdf5_char_array(epo['title'])

        # 5. 读取数据矩阵 X, y, t
        #    测试X 为 (trials, ch, time)
        X = epo['x'][()]       # numpy array
        y = epo['y'][()]       # 测试集y为空
        t = epo['t'][()]       # ms

    # 6. 转换为 MNE 需要的格式
    data = X / 1e6                              # uV → V
    times = t.squeeze() / 1000.0                # ms → s

    # 7. 创建 info
    info = mne.create_info(ch_names=clab, sfreq=fs, ch_types='eeg')
    info['description'] = title
    info['device_info'] = {'type': 'EEG', 'Model': 'BrainAmp'}
    # 8. 导入 montage 坐标

    # a. 构建 ch_pos
    ch_pos = {name: coord for name, coord in zip(clab, pos3d)}

    # b. 生成 DigMontage 并挂到 info
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')

    # c. 在创建完 info 后，调用
    info.set_montage(montage)

    # 9. 构造 event
    n_trials, _, n_times = data.shape
    trial_starts = np.arange(n_trials) * n_times

    answer_sheet_path = Path(folder_dir) / "Track3_Answer Sheet_Test.xlsx"
    df = pd.read_excel(answer_sheet_path, sheet_name='Track3', engine='openpyxl')

    if 2 * sub_i >= len(df.columns):
        raise IndexError(f"Subject {sub_i} 对应的 Answer Sheet 列不存在")
    y = df.iloc[2:, 2 * sub_i]  # 取 Data_Sample 中 True Label 下的所有行
    event_values = np.array(y.tolist()) - 1  # event_id 从 0 开始

    events = np.column_stack((trial_starts,
                              np.zeros(n_trials, dtype=int),
                              event_values))

    # 10. 创建 EpochsArray
    sub_epochs = mne.EpochsArray(data=data,
                                 info=info,
                                 tmin=times[0],
                                 events=events,
                                 event_id=event_id)

    return sub_epochs


def load_train_and_validation(folder_dir, sub_i, event_id, key='epo_train'):
    """
    加载训练集或验证集数据并构造 MNE EpochsArray。

    从指定目录加载训练或验证数据的 MAT 文件，读取通道名、坐标、数据和事件标签，
    并返回符合 MNE 格式的 EpochsArray 对象。

    Args:
        folder_dir (str or Path): 数据所在目录（Training set 或 Validation set）。
        sub_i (int): 被试编号（从 1 开始）。
        event_id (dict): 事件标识映射字典。
        key (str): MAT 文件中 epoch 数据的键，默认为 'epo_train'，验证集为 'epo_validation'。

    Returns:
        mne.EpochsArray: 加载并转换后的训练或验证集 Epochs 对象。
    """
    matfile_path = Path(folder_dir) / f"Data_Sample{sub_i:02d}.mat"

    mat_data = sio.loadmat(matfile_path)

    # 1. channel name and position
    epo = mat_data[key]
    mnt = mat_data['mnt']

    clab = [str(np.squeeze(label)) for label in mnt[0, 0]['clab'].squeeze()]
    pos3d = mnt[0, 0]['pos_3d']

    # 2. fs
    fs = np.squeeze(epo[0, 0]['fs'])

    # 3. title
    title = str(np.squeeze(epo[0, 0]['title']))

    # 4. x (times, chs, trials)
    X = epo[0, 0]['x']

    # 5. y (event in one-hot encode, class_num*trials)
    y = epo[0, 0]['y']

    # 6. t (epoch time)
    t = epo[0, 0]['t']  # ms

    # 7. 转换为 MNE 需要的格式
    data = X / 1e6                              # uV → V
    data = np.transpose(data, (2, 1, 0))        # convert to trials*chs*times
    times = t.squeeze() / 1000.0                # ms → s

    # 8. 创建 info
    info = mne.create_info(ch_names=clab, sfreq=fs, ch_types='eeg')
    info['description'] = title
    info['device_info'] = {'type': 'EEG', 'Model': 'BrainAmp'}

    # 9. 导入 montage 坐标
    # a. 把 pos_3d 转成 (n_ch, 3)
    pos3d = pos3d.T   # 64×3

    # b. 构建 ch_pos
    ch_pos = {name: coord for name, coord in zip(clab, pos3d)}

    # c. 生成 DigMontage 并挂到 info
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')

    # d. 在创建完 info 后，调用
    info.set_montage(montage)

    # 10. 构造 event
    n_trials, _, n_times = data.shape
    trial_starts = np.arange(n_trials) * n_times
    event_values = np.argmax(y, axis=0)  # one-hot encoded to normal marker

    events = np.column_stack((trial_starts,
                              np.zeros(n_trials, dtype=int),
                              event_values))

    # 11. 创建 EpochsArray
    sub_epochs = mne.EpochsArray(data=data,
                                 info=info,
                                 tmin=times[0],
                                 events=events,
                                 event_id=event_id)

    return sub_epochs


def raw_to_fif(raw_data_dir=RAW_DATA_DIR,
               event_id=EVENT_ID,
               folder_name_list=FOLDER_NAME_LIST,
               save_dir=SAVE_DIR):
    """
    遍历原始数据目录，处理所有子目录和被试，将 Epochs 数据保存为 FIF 文件。

    对 Training set、Validation set 和 Test set 中的每个被试数据分别调用
    load_train_and_validation 或 load_test，并将结果保存到指定目录。

    Args:
        raw_data_dir (str or Path): 原始数据根目录。
        event_id (dict): 事件标识映射字典。
        save_dir (str or Path): 保存输出 FIF 文件的目录。

    Returns:
        None
    """
    # 1. 在模块最顶端或 main 函数里，配置 logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)

    raw_data_dir = Path(raw_data_dir)
    save_dir = Path(save_dir)

    for folder_name in folder_name_list:

        logger.info(f"开始处理目录 “{folder_name}”")
        folder_dir = raw_data_dir / folder_name
        save_folder_dir = save_dir / folder_name
        save_folder_dir.mkdir(parents=True, exist_ok=True)

        for sub_i in range(1, 16):

            logger.info(f"正在处理 {folder_name} 下的 subject {sub_i}")

            if folder_name == "Training set":
                sub_epochs = load_train_and_validation(folder_dir, sub_i, event_id, key='epo_train')
            elif folder_name == "Validation set":
                sub_epochs = load_train_and_validation(folder_dir, sub_i, event_id, key="epo_validation")
            elif folder_name == "Test set":
                sub_epochs = load_test(folder_dir, sub_i, event_id, key="epo_test")

            save_path = save_folder_dir / f"sub{sub_i-1}_epo.fif"
            sub_epochs.save(save_path, overwrite=True)
            logger.info(f"已保存：{save_path}")


if __name__ == "__main__":
    raw_to_fif()
