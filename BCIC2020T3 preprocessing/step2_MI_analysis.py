# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 21:11:52 2025

@author: Fujie
"""

#%% 1. 设置工作地址

import os
from pathlib import Path
current_path = Path.cwd()
print("当前路径为：", current_path)

#%% 2. 导入包
from itertools import product
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

from sklearn.base import BaseEstimator, TransformerMixin

from utility.feature_MI import (cal_one_feature_extraction_MI,
                                build_mi_summaries,
                                convert_mi_nat_to_bit,
                                IC_TYPES)
from utility.plot_func import (set_global_plot_style,
                               plot_IC_bandwise_MI)
from config.BCIC2020Track3_config import Config

#%%% 3. 设置预处理参数

n_jobs=10 #设置并行核数

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
output_dir=Path('./results/MI')
output_dir.mkdir(parents=True, exist_ok=True)

#%%


class PowerFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ave_power=np.var(X, axis=-1)
        log_power = np.log10(ave_power)
        return log_power 

feature_extractor=PowerFeatureTransformer()  


for timewin_name, BG_name, sub_i in product(timewin_name_list, BG_name_list, sub_list):
    IC_csv_path=data_dir / path_list[3] / f'sub{sub_i}'/f'sub{sub_i}_iclabels.csv'

    IC_filename = f'sub{sub_i}_{timewin_name}_{BG_name}_raw_ic_epo.fif'
    
    IC_folder_list = [
        data_dir / path_list[6] / data_folder / f"sub{sub_i}"
        for data_folder in data_folder_list
    ]
    
    feature_name=feature_extractor.__class__.__name__
    output_csv_path =  output_dir/feature_name/f"sub{sub_i}_{timewin_name}_{BG_name}_IC_MI.csv"
    cal_one_feature_extraction_MI(
        IC_folder_list = IC_folder_list,
        IC_data_name = IC_filename,
        IC_csv_path = IC_csv_path,
        output_csv_path =  output_csv_path,
        feature_extractor = feature_extractor,
        random_state = random_state,
        n_jobs= n_jobs,
        is_zscore = True,
    )

build_mi_summaries(
    timewin_name_list=timewin_name_list,
    BG_name_list=BG_name_list,
    sub_list=sub_list,
    mi_csv_dir=Path('./results/MI/PowerFeatureTransformer'),
    filename_pattern= "sub{sub}_{timewin}_{band}_IC_MI.csv",
    agg= "median")


#%%
# 设置全局图片
set_global_plot_style(width_cm = 8,
                      height_cm = 6,
                      font_family = "Times New Roman",
                      font_size = 5,
                      dpi = 600,
                      file_type='png',
                      )

# feature_name=feature_extractor.__class__.__name__
feature_name="PowerFeatureTransformer"

plot_dir=output_dir/'report_drawing'
plot_dir.mkdir(parents=True, exist_ok=True)  


df=pd.read_csv(output_dir/ feature_name / 'MI_total.csv')
df = convert_mi_nat_to_bit(df)

plot_IC_bandwise_MI(df=df,
                    timewin_name_list=timewin_name_list,
                    IC_types_list=IC_TYPES+['total'],
                    BG_name_list=BG_name_list,
                    save_dir = plot_dir/feature_name)


    
    
    
