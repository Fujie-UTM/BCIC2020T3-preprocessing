# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 21:34:35 2025

@author: Fujie
"""

#%% 1. 设置工作地址

import os
from pathlib import Path
current_path = Path.cwd()
print("当前路径为：", current_path)

#%% 2. 导入包
import sys
import shutil
import mne
import json
import time
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from joblib import parallel_backend

from utility.BCIC2020Track3 import raw_to_fif
from utility.preprocess import *
from config.BCIC2020Track3_config import Config
#%%% 3. 设置预处理参数

n_jobs=1 #设置并行核数

CONFIG=Config()

random_state=CONFIG.random_state #随机数种子
np.random.seed(random_state)

sub_list = CONFIG.sub_list
path_list = CONFIG.path_list
data_folder_list = CONFIG.data_folder_list
BG_name_list = CONFIG.BG_name_list
BG_list = CONFIG.BG_list
timewin_name_list = CONFIG.timewin_name_list
timewin_list = CONFIG.timewin_list
artifact_type = CONFIG.artifact_type
prob_critera_ic = CONFIG.prob_critera_ic
NF_freq_list = CONFIG.NF_freq_list
num_chs = CONFIG.num_chs
event_id = CONFIG.event_id

data_dir=Path('./data')
for path in path_list:
    folder_path=data_dir / path
    folder_path.mkdir(parents=True, exist_ok=True)
    
#%%    
raw_data_dir = data_dir/path_list[0]/'Track#3 Imagined speech classification'
fif_save_dir = data_dir/path_list[1]

raw_to_fif(raw_data_dir= raw_data_dir ,
           event_id = event_id,
           folder_name_list = data_folder_list,
           save_dir = fif_save_dir)

#%% 4. 共参考
# 将训练集和测试集设为共参考
for data_folder in data_folder_list:
    for sub_i in sub_list:
        # 4.1. 读取上级文件夹，被试fif文件
        input_dir = data_dir / path_list[1] / data_folder #读取训练集数据         

        sub_epochs = mne.read_epochs(input_dir / f"sub{sub_i}_epo.fif")   
        
        output_dir = data_dir / path_list[2] / data_folder 
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sub_epochs_CAR = sub_epochs.copy().set_eeg_reference(ref_channels='average')
        sub_epochs_CAR.save(output_dir / f"sub{sub_i}_epo.fif", overwrite=True)

#%% 5.ICA      

# ICA train
for sub_i in sub_list:     
    # 5.1.1 读取前一级数据  
    input_dir = data_dir / path_list[2] / data_folder_list[0]   

    sub_epochs_CAR_train = mne.read_epochs(input_dir / f'sub{sub_i}_epo.fif')  

    output_dir = data_dir / path_list[3] / f'sub{sub_i}'
    
    #训练ICA
    ica, ic_labels, ic_probas =  sub_ica_train(epochs = sub_epochs_CAR_train,
                                               fname = f'sub{sub_i}',
                                               NF_freq_list = NF_freq_list,
                                               out_dir = output_dir,
                                               random_state=None,
                                               n_jobs = n_jobs,
                                               need_CAR = False,
                                               plot_ic = True)
    
#%%6. 时间段提取
for sub_i in sub_list:
    for data_folder in data_folder_list:
        
        input_dir = data_dir / path_list[2] / data_folder   
        
        sub_epochs_raw =  mne.read_epochs(input_dir / f'sub{sub_i}_epo.fif')
                    
        output_dir =  data_dir / path_list[4] / data_folder /f'sub{sub_i}' 
        
        sub_time_split(epochs = sub_epochs_raw,
                       fname = f'sub{sub_i}',
                       output_dir = output_dir,
                       timewin_list = timewin_list,
                       timewin_name_list = timewin_name_list, 
                       plot_comparison=True,
                       n_jobs= n_jobs)
        
 #%% 7. 频率截取

for sub_i in sub_list:
    for data_folder in data_folder_list:
        for timewin_name in timewin_name_list:        
            
            input_dir = data_dir / path_list[4] / data_folder / f'sub{sub_i}'
            
            sub_epochs_raw_t = mne.read_epochs(input_dir / f'sub{sub_i}_{timewin_name}_epo.fif')
            
            output_dir =  data_dir / path_list[5] / data_folder /f'sub{sub_i}' 

            sub_frequency_split(epochs = sub_epochs_raw_t,
                                fname = f'sub{sub_i}_{timewin_name}',
                                output_dir = output_dir,
                                NF_freq_list = NF_freq_list,
                                BG_list = BG_list,
                                BG_name_list = BG_name_list, 
                                plot_comparison=True,
                                n_jobs=n_jobs)
#强制恢复输出            
sys.stdout = sys.__stdout__             
#%% 5.ICA apply  

for sub_i in sub_list:
        
    ica_dir = data_dir / path_list[3] / f'sub{sub_i}'   
    
    ica_df = pd.read_csv(ica_dir / f'sub{sub_i}_iclabels.csv')
    sub_ica = read_ica(ica_dir / f'sub{sub_i}-ica.fif')
    exclude_idx_list, exclude_label_list = get_excluded_ics(df = ica_df, 
                                                            artifact_type=artifact_type, 
                                                            prob_thresh=prob_critera_ic)
    for data_folder in data_folder_list:        
        for timewin_name in timewin_name_list:     
            for BG_name in BG_name_list:

                input_dir = data_dir / path_list[5] / data_folder / f'sub{sub_i}'
                
                sub_epochs_raw_f = mne.read_epochs(input_dir / f'sub{sub_i}_{timewin_name}_{BG_name}_epo.fif')
                
                output_dir =  data_dir / path_list[6] / data_folder /f'sub{sub_i}' 
                
                sub_ica_apply(epochs = sub_epochs_raw_f,
                              ica = sub_ica,
                              fname = f'sub{sub_i}_{timewin_name}_{BG_name}',
                              exclude_label_list = exclude_label_list,
                              exclude_idx_list = exclude_idx_list,
                              output_dir = output_dir,
                              save_ica = True,
                              plot_comparison = True,
                              n_jobs = n_jobs)
                                                        
                

#%% 将fif转化为h5py,聚类

ica_type_list=['raw', 'raw_ic', 'clean', 'clean_ic', 'excluded', 'excluded_ic']
        
for ica_type in ica_type_list:
    for data_folder in data_folder_list:
        for timewin_name in timewin_name_list:     
            #数据太大，我们只做gamma和全频段
            for BG_name in BG_name_list:

                output_dir =  data_dir / path_list[7] / data_folder / timewin_name/ BG_name/ ica_type
                output_dir.mkdir(parents=True, exist_ok=True)
                
                for sub_i in sub_list:
                    input_dir =  data_dir / path_list[6] / data_folder /f'sub{sub_i}' 
 
                    filename = f'sub{sub_i}_{timewin_name}_{BG_name}_{ica_type}_epo'
                    fif_to_h5_epoch(fif_file = input_dir / f'{filename}.fif', 
                                    h5_file = output_dir / f'{filename}.h5', 
                                    overwrite=True, 
                                    target_channels=num_chs)
                    

        
        
        

        
        