# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 19:02:51 2025

@author: Fujie
"""

from pathlib import Path
import os
import mne
import numpy as np
import scipy.io as sio

def convert_BCI2020track3_to_fif(
    input_dir='./data',
    output_dir='./data/1_epoch_data',
    folder_name_list=['Training set', 'Validation set'],
    subject_range = range(1, 16),
    track_subfolder='Track#3 Imagined speech classification'
):
    """将 2020 BCI Competition Track 3 语音想象的 .mat 数据批量转换为 MNE EpochsArray 并保存为 _epo.fif。

    本函数会遍历指定根目录下的 Track#3 Imagined speech classification 数据，
    对 Training set 和 Validation set 中的所有被试进行以下操作：
      1. 读取 .mat 文件中的 epo 数据和对应的元信息（通道名、采样率、时间、事件等）；
      2. 将信号单位从 μV 转换为 V，调整数据维度；
      3. 构建 MNE Info 和 EpochsArray 对象，并附加 EEG montage；
      4. 将结果保存为 _epo.fif，可选注释掉的 FCz 参考电极和平均参考代码保留；
      5. 输出保存路径及相关信息。

    Args:
        root (str | pathlib.Path): 原始 .mat 数据所在的根目录路径。
        output_subdir (str): 保存输出 _epo.fif 文件的子目录名，默认 '1_epoch_data'。
        folder_name_list (Sequence[str]): 参与转换的数据集文件夹名称列表，
            默认为 ('Training set', 'Validation set')。
        subject_range (Sequence[int]): 被试编号范围，默认 1-15。
        track_subfolder (str): 指定 Track#3 数据所在的子目录名，默认
            'Track#3 Imagined speech classification'。

    Returns:
        None

    Raises:
        FileNotFoundError: 未找到指定的 Track#3 子目录或数据集文件夹时抛出。
        OSError: 创建输出目录或保存 _epo.fif 文件时发生 I/O 错误。
        Exception: 在读取 .mat 或转换过程中出现其他异常时抛出。
    """
    input_dir = Path(input_dir)
    print(f"数据目录路径：{input_dir}")

    # 输出基础目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for folder_name in folder_name_list:
        # 构建当前数据集路径
        data_folder = input_dir / track_subfolder / f"{folder_name}/"
        if not data_folder.exists():
            raise FileNotFoundError(f"未找到数据文件夹：{data_folder}")

        for i in subject_range:
            # 拼接 mat 文件名
            if i <= 9:
                mat_name = f"Data_Sample0{i}.mat"
            else:
                mat_name = f"Data_Sample{i}.mat"
            matfile_path = data_folder / mat_name
            print(f"Processing {matfile_path.name} ...")

            # 读取 .mat 数据
            mat_data = sio.loadmat(matfile_path)
            key = 'epo_train' if folder_name == 'Training set' else 'epo_validation'
            epo = mat_data[key]
            mnt = mat_data['mnt']

            # 1. clab
            clab_obj = epo[0,0]['clab']
            clab = np.array(np.zeros(clab_obj.shape), dtype=str)
            for j in range(clab.shape[1]):
                clab[0,j] = str(np.squeeze(clab_obj[0,j]))
            clab = np.squeeze(clab).tolist()
            del clab_obj

            # 2. fs
            fs = float(np.squeeze(epo[0,0]['fs']))

            # 3. title
            title = str(np.squeeze(epo[0,0]['title']))

            # 4. file
            file = str(np.squeeze(epo[0,0]['file']))

            # 5. x (t * ch * trial)
            X = epo[0,0]['x']

            # 6. y (one-hot encoded events)
            y = epo[0,0]['y']

            # 7. t (epoch 时间)
            t = epo[0,0]['t']

            # 8. className
            className_obj = epo[0,0]['className']
            className = np.array(np.zeros(className_obj.shape), dtype=str)
            for j in range(className.shape[1]):
                className[0,j] = str(np.squeeze(className_obj[0,j]))
            className = np.squeeze(className).tolist()
            del className_obj

            # 转换数据到 EpochsArray
            data = X / 1e6  # 单位 μV to V
            data = np.transpose(data, (2, 1, 0))  # 转为 trials * ch * time
            time = t / 1000  # ms to s

            # 创建 Info
            info = mne.create_info(ch_names=clab, sfreq=fs, ch_types='eeg')
            info['description'] = title
            info['device_info'] = {'type': 'EEG', 'Model': 'BrainAmp'}

            event_id = {'Hello': 0, 'Help me': 1, 'Stop': 2, 'Thank you': 3, 'Yes': 4}
            epoch_num = data.shape[0]
            epoch_duration = data.shape[2]
            event_value = np.argmax(y, axis=0)
            events = np.column_stack((
                np.arange(0, epoch_num * epoch_duration, epoch_duration),
                np.zeros(epoch_num, dtype=int),
                event_value
            ))

            sub_epochs = mne.EpochsArray(
                data, info, tmin=time[0, 0], events=events, event_id=event_id
            )

            sub_epochs.set_montage('standard_1020')

            # 构建保存路径并保存 fif 文件
            save_path = Path(output_dir / folder_name)
            save_path.mkdir(parents=True, exist_ok=True)
            savename = save_path / f"sub{i-1}_epo.fif"
            sub_epochs.save(savename, overwrite=True)
            print(f"  → Saved to {savename}")
