# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 21:54:29 2025

@author: Fujie
"""

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Config:

    random_state=99
    num_sub=15   #被试数目
    sub_list=np.arange(num_sub) #被试编号列表
    num_chs = 64
    sfreq = 256
    
    NF_freq_list= [60]
    BG_name_list=['Low_Delta','Delta','Theta','Alpha','Beta','Gamma', 'Full']
    BG_list=[[0, 4],[0.5, 4], [4, 8], [8, 16], [16, 30], [30, 125], [0.5, 125]]
    
    timewin_name_list=['-500_0','0_2000','-500_2000']
    timewin_list=[[-0.5, 0], [0, 2], [-0.5, 2]]
    
    prob_critera_ic=0.9 #icalabel 自动剔除判别概率
    artifact_type=['muscle artifact','eye blink','heart beat','line noise',
                   'channel noise'] #剔除的伪迹种类 20241203改
    
    path_list = ['0_raw_data','1_epoch_data','2_CARrefered', '3_ica_train','4_time_extract',
                 '5_filtered', '6_ica_apply', '7_h5py',]
    
    data_folder_list=['Training set', 'Validation set', 'Test set']
    event_id={'Hello': 0, 'Help me': 1, 'Stop': 2, 'Thank you': 3, 'Yes': 4}


    
    # 机器学习参数
    num_features=20                 #15项独立成分, 至少要保证8个成分，多于12个成分效果有限
    cv_n_splits=5                   #交叉验证折数
    cv_n_repeats=2                  #交叉验证重复次数
    cv_n_repeats_stat=20            #训练集用于展示统计结果的重复次数