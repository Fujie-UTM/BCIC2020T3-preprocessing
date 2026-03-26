# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 20:19:17 2025

@author: Fujie
"""
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
from PIL import Image
from mne.preprocessing import ICA, read_ica
from mne_icalabel import label_components

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import (GridSearchCV, 
                                     RepeatedStratifiedKFold, 
                                     cross_validate)
from sklearn.metrics import (balanced_accuracy_score, 
                             confusion_matrix, 
                             ConfusionMatrixDisplay,
                             f1_score,
                             cohen_kappa_score)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC

from scipy.stats import wilcoxon, binom
from statsmodels.stats.multitest import multipletests
import numpy as np
from pathlib import Path
import mne
import pandas as pd
import matplotlib.pyplot as plt
from mne.preprocessing import ICA, read_ica
from mne_icalabel import label_components
import numpy as np
import mne
import h5py
from typing import Union, Optional, List
from contextlib import redirect_stdout
#%%
CONFIG = {  'notch_iir_params': dict(order=4, ftype='butter') ,
            'BP_iir_params': dict(order=4, ftype='butter'),
            'wave_plot_n_chs': 10,
            'wave_plot_n_epochs': 10,
            'wave_plot_scalings': 400e-6
}




def plot_waveform_spectrum(
    epochs,
    output_dir,
    title: str = 'postprocess',
    n_jobs: int = -1,
):
    """
    绘制并保存 EEG epochs 的时域波形图和频谱图。

    参数
    ----------
    epochs : mne.Epochs
        要绘制的 epochs 对象。
    output_dir : str or Path
        输出目录，图像文件将保存在此路径下。
    title : str, default 'postprocess'
        图像标题前缀，也将用于文件名。
    n_jobs : int, default -1
        并行计算 PSD 时使用的作业数，-1 表示使用所有可用核。

    返回
    -------
    None
    """

    # 确保 output_dir 存在，不存在则递归创建
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 强制使用 matplotlib 后端，以便在无 GUI 环境中也能正常绘图
    with mne.viz.use_browser_backend("matplotlib"):
        # 1) 绘制时域波形
        #    - events=False: 不在波形上叠加事件标记
        #    - scalings: 每个通道的 y 轴缩放比例
        #    返回一个 matplotlib.figure.Figure 对象
        fig_wave = epochs.plot(picks='all',
            n_channels=CONFIG['wave_plot_n_chs'],
            n_epochs=CONFIG['wave_plot_n_epochs'],
            title=title,
            events=False,
            scalings=CONFIG['wave_plot_scalings']
        )
        # 保存成 PNG，并收紧边距
        fig_wave.savefig(output_dir / f"{title}_waveform.png", bbox_inches='tight')
        plt.close(fig_wave)

        # 2) 计算功率谱密度（Power Spectral Density）
        #    默认使用 Welch 方法，可通过额外参数如 fmin、fmax、n_fft 来定制
        psd = epochs.compute_psd(picks= 'all', 
                                 n_jobs=n_jobs)

        # 绘制 PSD（对数坐标 X 轴）
        #    - amplitude=False: 绘制功率，而非振幅谱
        #    - xscale='log': 频率轴使用对数刻度
        fig_spec = psd.plot(picks ='all',
            amplitude=False,
            xscale='log'
        )
        fig_spec.savefig(output_dir / f"{title}_spectrum.png", bbox_inches='tight')
        plt.close(fig_spec)



def ica_preparation_epoch(epochs, NF_freq_list, n_jobs, need_CAR=False):
    """
    对 CAR 参考下的 epochs 做工频陷波（notch）+ 带通预滤波，返回处理后的 epochs 对象。

    参数
    ----------
    epochs : mne.Epochs
        原始的 EEG epochs 数据。
    NF_freq_list : list of float or list of list
        需要陷波的频率列表，例如 [50, 100]，或 [[49, 51], [99, 101]]。
    n_jobs : int
        并行处理时使用的进程数。
    need_CAR : bool, default False
        是否先做 Common Average Reference（平均参考）。

    返回
    -------
    epochs_prepared : mne.Epochs
        经 notch + 带通滤波处理后的 epochs。
    """
    # 1. 复制原 epochs，防止修改到外部对象
    epochs_temp = epochs.copy()

    # 2. 如果需要，先做平均参考。平均参考能减少空间共模干扰，有利于后续 ICA 分解。
    if need_CAR:
        epochs_temp.set_eeg_reference(ref_channels='average')

    # 3. 从 epochs 中提取数据矩阵，形状为 (n_epochs, n_channels, n_times)
    data = epochs_temp.get_data(copy=True)
    sfreq = epochs_temp.info['sfreq']

    # 4. 对每个指定频率或频段依次做 notch 滤波以去除工频干扰
    for NF_freq in NF_freq_list:
        # mne.filter.notch_filter 支持多频点一次滤波，也可以多次调用
        data = mne.filter.notch_filter(x=data,
                                       Fs=sfreq,
                                       freqs=NF_freq,
                                       trans_bandwidth=2, # 过渡带宽度 ±2 Hz
                                       method='iir',      # IIR 实现
                                       phase='zero',      # 零相位，避免相位失真
                                       iir_params=CONFIG['notch_iir_params'],  # IIR 设计参数
                                       n_jobs=n_jobs,)

    # 5. 将滤波后的数据写回 epochs 对象
    epochs_temp._data = data

    # 6. 对预处理后的数据做 1-100 Hz 带通滤波（ICA 通常在此频段内训练效果最佳）
    epochs_prepared = epochs_temp.filter(l_freq=1,
                                         h_freq=100,
                                         picks='eeg',
                                         n_jobs=n_jobs)

    return epochs_prepared

        

def sub_ica_train(
    epochs,
    fname: str,
    NF_freq_list,
    out_dir,
    random_state=None,
    n_jobs: int = -1,
    need_CAR: bool = False,
    plot_ic: bool = False
):
    """
    对给定 epochs 拟合 ICA 并保存相关产物：
      1. 原始 ICA 源信号（Epochs 格式，可选保存）
      2. 每个 IC 的 ICLabel “Type” 与“Proba”表格
      3. 训练好的 ICA 模型文件（.fif）
      4. （可选）每个 IC 的属性图

    参数
    ----------
    epochs : mne.Epochs
        输入的分段数据。
    fname : str
        用于结果文件命名前缀（如被试 ID）。
    NF_freq_list : list of float or list of (low, high) pairs
        需要做 notch 滤波的频率或频段列表，例如 [50, 100] 或 [(49,51), (99,101)]。
    out_dir : str or Path
        结果输出目录。
    random_state : int | None
        随机种子，保证 ICA 拟合可复现。默认为 None。
    n_jobs : int, default -1
        并行作业数，-1 表示使用全部可用核心。
    suffix : str, default '-ica'
        ICA 模型文件名后缀（不含“.fif”）。实际保存时会自动加上 `.fif`。
    need_CAR : bool, default False
        是否先对数据做 Common Average Reference。
    plot_ic : bool, default False
        是否绘制并保存每个成分的属性图。

    返回
    -------
    ica : mne.preprocessing.ICA
        拟合完成的 ICA 对象。
    labels : list of str
        ICLabel 给出的每个成分类别标签。
    proba : ndarray, shape (n_components, n_classes)
        每个成分对应各类别的预测概率。
    """

    # —— 1. 创建输出目录
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # —— 2. 拷贝 epochs，
    epochs_temp = epochs.copy()

    # —— 3. 预处理：notch + 带通
    # 注意：参数名应与函数定义保持一致
    epochs_prepared = ica_preparation_epoch(
        epochs_temp,
        NF_freq_list=NF_freq_list,
        n_jobs=n_jobs,
        need_CAR=need_CAR   # 已在上面处理过 CAR
    )

    # —— 4. 构建并拟合 ICA：组件数设为通道数减 1
    n_ch = len(epochs_prepared.ch_names)
    ica = ICA(
        n_components=n_ch - 1,
        method='infomax',
        fit_params=dict(extended=True),
        max_iter='auto',
        random_state=random_state
    )
    ica.fit(epochs_prepared)

    # —— 5. 获取原始 ICA 源信号（Epochs 对象）
    raw_sources = ica.get_sources(epochs_prepared)

    # —— 6. 使用 ICLabel 给每个成分打标签和概率分数
    ic_label_dict = label_components(epochs_prepared, ica, method='iclabel')
    labels = ic_label_dict['labels']
    proba = ic_label_dict['y_pred_proba']

    # —— 7. 保存 ICLabel 结果到 CSV
    # 合并 IC 名称、标签、概率
    IC_names = raw_sources.ch_names
    df_ic = pd.DataFrame({
        'IC_name': IC_names,
        'Type': labels,
        'Proba': proba
    })
    ic_csv = out_dir / f'{fname}_iclabels.csv'
    df_ic.to_csv(ic_csv, index=False, encoding='utf-8')

    # —— 8. 保存 ICA 模型（.fif）
    ica_path = out_dir / f'{fname}-ica.fif'
    ica.save(ica_path, overwrite=True)


    # —— 9. （可选）绘制并保存每个成分的属性图
    if plot_ic:
        fig_dir = out_dir / f"{fname}_IC_properties"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for IC_idx, IC_name in enumerate(IC_names):
            figs = ica.plot_properties(epochs_prepared, picks=IC_idx)
            fig = figs[0]
            fig.savefig(fig_dir / f'{fname}_{IC_name}.png', bbox_inches='tight')
            plt.close(fig)

    return ica, labels, proba

def get_excluded_ics(df, artifact_type, prob_thresh):
    """
    从 DataFrame 中筛选要剔除的 ICA 成分。

    Args:
        df (pd.DataFrame): 包含列 ['IC_name', 'Type', 'Proba'] 的 DataFrame。
        artifact_type (list[str]): 被认为是伪迹的标签类型（如 ['eye blink']）。
        prob_thresh (float): 判别概率阈值（如 0.9）。

    Returns:
        tuple: (exclude_idx_list, exclude_label_list)
            - exclude_idx_list: ICA编号（int）
            - exclude_label_list: 对应的标签
    """
    exclude_idx_list = []
    exclude_label_list = []

    # 条件筛选（Proba ≥ 阈值 且 Type 属于指定类别）
    filtered_df = df[(df['Proba'] >= prob_thresh) & (df['Type'].isin(artifact_type))]

    for _, row in filtered_df.iterrows():
        # 提取 ICA 编号数字（如 ICA004 → 4）
        ic_index = int(row['IC_name'].replace('ICA', ''))
        exclude_idx_list.append(ic_index)
        exclude_label_list.append(row['Type'])

    return exclude_idx_list, exclude_label_list


def sub_ica_apply(
    epochs,
    ica,
    fname,
    exclude_label_list,
    exclude_idx_list,
    output_dir=None,
    save_ica=False,
    plot_comparison=True,
    n_jobs=-1
):
    """根据给定的 IC 索引列表，剔除对应的伪迹成分并重构信号，同时保存各类结果。

    本函数执行流程：
      1. 创建输出目录。
      2. 标准化要剔除的成分索引列表。
      3. 复制并准备三个 ICA 对象：原始、清理后、仅保留伪迹成分。
      4. 将剔除信息写入文本日志。
      5. 分别重构：
         - epochs_clean：剔除伪迹后的信号
         - epochs_excluded：仅含被剔除成分的信号
      6. 提取三种情形下的独立成分时段：
         - epochs_ICcomp_raw：原始所有成分
         - epochs_ICcomp_clean：清理后剩余成分
         - epochs_ICcomp_excluded：仅被剔除成分
      7. 将上述所有 Epochs 保存为 FIF 文件；可选保存清理后 ICA 模型。
      8. 可选绘制波形／频谱对比图。

    Args:
        epochs (mne.Epochs): 原始分段数据（未应用 ICA）。
        ica (mne.preprocessing.ICA): 已拟合的 ICA 对象。
        fname (str): 输出文件名前缀（如被试 ID）。
        exclude_label_list (List[str]): 与 exclude_idx_list 一一对应的标签名称列表。
        exclude_idx_list (int or List[int]): 要剔除的 IC 索引或列表。
        output_dir (str or pathlib.Path, optional): 保存结果的目录，None 则不保存。默认 None。
        save_ica (bool): 是否将清理后的 ICA 对象保存到磁盘。默认 False。
        plot_comparison (bool): 是否绘制重构后信号对比图。默认 True。
        n_jobs (int): 并行作业数，传给绘图函数，-1 表示全部 CPU。默认 -1。

    Returns:
        tuple:
            epochs_clean (mne.Epochs): 剔除伪迹后的重构 epochs。
            epochs_ICcomp_raw (mne.Epochs): 原始 ICA 各成分 epochs。
            epochs_ICcomp_clean (mne.Epochs): 清理后 ICA 各成分 epochs。
            epochs_excluded (mne.Epochs): 仅被剔除成分的重构 epochs。
            epochs_ICcomp_excluded (mne.Epochs): 仅被剔除成分的源信号 epochs。

    Raises:
        OSError: 输出目录创建或文件写入失败。
        Exception: ICA 重构或保存过程出错。
    """
    from pathlib import Path

    # 1. 创建并准备输出目录（如果指定）
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 标准化 exclude_idx_list 为列表
    if isinstance(exclude_idx_list, int):
        exclude_idx_list = [exclude_idx_list]

    # 3. 复制 ICA 对象，分别用于：
    #    - ica_raw：保留原始未剔除状态
    #    - ica_clean：标记剔除列表，重构“清理后”信号
    #    - ica_excluded：标记保留列表，仅重构伪迹贡献
    ica_raw      = ica.copy()
    ica_clean    = ica.copy()
    ica_excluded = ica.copy()

    # 将要剔除的成分索引添加到 clean ICA 的 exclude
    ica_clean.exclude.extend(exclude_idx_list)
    # 在 excluded ICA 中，剔除所有非伪迹成分，保留伪迹
    all_idx = list(range(ica_excluded.n_components_))
    ica_excluded.exclude = [i for i in all_idx if i not in exclude_idx_list]

    # 4. 保存剔除的 IC 信息到文本文件
    if output_dir is not None:
        info_lines = ["Rejected ICs (index \t label):"]
        for idx, label in zip(exclude_idx_list, exclude_label_list):
            info_lines.append(f"{idx}\t{label}")
        txt_path = output_dir / f"{fname}_rejected_ICs.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(info_lines))

    # 5. 应用 ICA 重构
    epochs_raw   = epochs.copy()  # 备份原始 epochs
    epochs_clean = epochs.copy()  # 用于清理后信号
    epochs_excluded = epochs.copy()  # 用于仅保留伪迹信号

    # 计算重构时的 PCA 分量数：保留通道数 - 1
    num_chs = len(epochs_clean.ch_names)
    # 清理后重构
    ica_clean.apply(epochs_clean, n_pca_components=num_chs-1)
    # 仅保留伪迹成分重构
    ica_excluded.apply(epochs_excluded, n_pca_components=num_chs-1)

    # 6. 提取各情形下的 IC 成分时段
    epochs_ICcomp_raw       = ica_raw.get_sources(epochs_raw).copy()
    # 清理后，挑出保留的成分
    epochs_ICcomp_clean     = (
        ica_clean.get_sources(epochs_clean)
                 .copy()
                 .pick(picks='misc', exclude='bads')
    )
    # 仅伪迹成分
    epochs_ICcomp_excluded = (
        epochs_ICcomp_raw.copy()
                          .pick(picks=exclude_idx_list)
    )

    # 7. 保存所有结果文件
    if output_dir is not None:
        # 重构前后 epochs
        epochs_raw.save(  output_dir / f"{fname}_raw_epo.fif",     overwrite=True)
        epochs_clean.save(output_dir / f"{fname}_clean_epo.fif",   overwrite=True)
        epochs_excluded.save(output_dir / f"{fname}_excluded_epo.fif", overwrite=True)
        # 各 IC 成分时段
        epochs_ICcomp_raw.save(     output_dir / f"{fname}_raw_ic_epo.fif",     overwrite=True)
        epochs_ICcomp_clean.save(   output_dir / f"{fname}_clean_ic_epo.fif",   overwrite=True)
        epochs_ICcomp_excluded.save(output_dir / f"{fname}_excluded_ic_epo.fif", overwrite=True)
        # 可选：保存清理后 ICA 模型
        if save_ica:
            ica_clean.save(output_dir / f"{fname}_clean-ica.fif", overwrite=True)

    # 8. 可选：绘制重构后信号的波形与频谱对比
    if plot_comparison and output_dir is not None:
        plot_waveform_spectrum(
            epochs=epochs_clean,
            output_dir=output_dir,
            title=f"{fname}",
            n_jobs=n_jobs
        )
        plot_waveform_spectrum(
            epochs=epochs_excluded,
            output_dir=output_dir,
            title=f"{fname}_excluded",
            n_jobs=n_jobs
        )

    return (
        epochs_clean,
        epochs_ICcomp_raw,
        epochs_ICcomp_clean,
        epochs_excluded,
        epochs_ICcomp_excluded
    )


        
        
        
def sub_time_split(
    epochs,
    fname,
    output_dir,
    timewin_list,
    timewin_name_list, 
    plot_comparison=False,
    n_jobs=-1
):
    """按指定的多个时间窗口对 epochs 数据进行裁剪并保存。

    对于给定的每个时间窗口，复制原始 epochs 对象，截取相应的时间段，
    按照窗口名称创建子目录，将截取后的 epochs 保存为 .fif 文件；
    如需对比还可调用波形谱绘制函数。

    Args:
        epochs (mne.Epochs): 原始的 epochs 对象，用于裁剪和保存。
        fname (str): 基础文件名（不含后缀），用于输出文件命名。
        output_dir (str or pathlib.Path): 根输出目录，函数会在其下
            根据窗口名称和 fname 创建子文件夹。
        timewin_list (List[Tuple[float, float]]): 时间窗口列表，每个元素
            为 (tmin, tmax) 二元组，单位为秒。
        timewin_name_list (List[str]): 与 timewin_list 对应的名称列表，
            用于命名各窗口对应的子目录及文件。
        plot_comparison (bool): 是否在保存后调用 plot_waveform_spectrum
            绘制波形谱对比图。默认 False，不绘制。
        n_jobs (int): 并行作业数，传递给 plot_waveform_spectrum。默认 -1，
            表示使用所有可用 CPU。

    Returns:
        None

    Raises:
        OSError: 若在创建目录或保存文件时发生 I/O 错误，则向上抛出异常。
    """
    # 遍历所有指定的时间窗口
    for timewin_i, timewin in enumerate(timewin_list):
        # 取出对应的窗口名称
        timewin_name = timewin_name_list[timewin_i]
        # 构建该窗口的输出路径
        output_dir = Path(output_dir) 
        # 递归创建目录（若已存在则忽略）
        output_dir.mkdir(parents=True, exist_ok=True)

        # 对 epochs 对象进行时间截取
        epochs_t = epochs.copy().crop(tmin=timewin[0],
                                      tmax=timewin[1])

        # 拼接输出文件名，例如 "fname_window_epo.fif"
        filename = f"{fname}_{timewin_name}_epo.fif"
        # 保存截取后的 epochs，若文件已存在则覆盖
        epochs_t.save(output_dir / filename, overwrite=True)

        # 如需绘制对比图，则调用外部函数
        if plot_comparison:
            plot_waveform_spectrum(
                epochs=epochs_t,
                output_dir=output_dir,
                title=f"{fname}_{timewin_name}",
                n_jobs=n_jobs,
            )



def sub_frequency_split(
    epochs,
    fname,
    output_dir,
    NF_freq_list,
    BG_list,
    BG_name_list, 
    plot_comparison=False,
    n_jobs=-1
):
    """对 epochs 数据进行陷波和带通滤波并保存到磁盘，记录滤波日志。

    本函数会先对输入的 epochs 对象依次执行多个陷波滤波（notch filter）
    以去除电源工频等干扰，然后对不同的带通频段执行滤波操作，
    并将每个频段处理后的 epochs 保存为 .fif 文件。所有滤波过程
    的标准输出会重定向到一个日志文件中，便于后续检查。

    Args:
        epochs (mne.Epochs): 待处理的原始 epochs 对象。
        fname (str): 基础文件名（不含后缀），用于生成输出文件名和日志文件名。
        output_dir (str or pathlib.Path): 根输出目录，函数会在此创建必要的子目录和日志文件。
        NF_freq_list (List[float]): 陷波滤波器中心频率列表，每个频率对应一次陷波操作。
        BG_list (List[Tuple[float, float]]): 带通滤波频段列表，每个元素为 (l_freq, h_freq)。
        BG_name_list (List[str]): 与 BG_list 对应的带通频段名称列表，用于命名每个输出文件。
        plot_comparison (bool): 是否在每次带通滤波后调用 plot_waveform_spectrum
            绘制波形谱对比图。默认 False，不绘制。
        n_jobs (int): 并行作业数，传递给 notch_filter 和 plot_waveform_spectrum。默认 -1，
            表示使用所有可用 CPU 核心。

    Returns:
        None

    Raises:
        OSError: 无法创建输出目录或打开日志文件时抛出。
        Exception: 在滤波或保存过程中若发生其他错误，则向上抛出异常。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs_NF = epochs.copy()
    
    # 开启滤波记录保存
    logname = output_dir / (fname + '_filter.txt')
    with open(logname, "w") as f, redirect_stdout(f):

        temple_data = epochs_NF.get_data(copy=True)
        sfreq = epochs_NF.info["sfreq"]
        for NF_freq in NF_freq_list:
            temple_data = mne.filter.notch_filter(
                x=temple_data,
                Fs=sfreq,
                freqs=NF_freq,
                trans_bandwidth=2,
                method='iir',
                phase='zero',
                iir_params=CONFIG['notch_iir_params'],
                n_jobs=n_jobs
            )
        epochs_NF._data = temple_data
    
        # 7.5 进行带通滤波
        for BG_name, BG_val in zip(BG_name_list, BG_list):
            epochs_BG = epochs_NF.copy()
            epochs_BG.filter(
                l_freq=BG_val[0],
                h_freq=BG_val[1],
                method='iir',
                iir_params=CONFIG['BP_iir_params'],
                phase='zero',
                n_jobs=n_jobs
            )
            
            filename = f"{fname}_{BG_name}_epo.fif"
            epochs_BG.save(output_dir / filename, overwrite=True)
                        
            if plot_comparison:
                plot_waveform_spectrum(
                    epochs=epochs_BG,
                    output_dir=output_dir,
                    title=f"{fname}_{BG_name}",
                n_jobs=n_jobs,
            )


def fif_to_h5_epoch(fif_file, h5_file, overwrite=False, target_channels=None):
    """
    将单个 MNE .fif epochs 文件转换成 HDF5 格式。

    Args:
        fif_file (str or Path): 输入的 .fif 文件路径（应以 _epo.fif 结尾）。
        h5_file (str or Path): 输出的 .h5 文件完整路径。
        overwrite (bool): 如果目标文件已存在，是否覆盖。默认为 False。
        target_channels (int or None): 
            如果指定，且原始通道数少于该值则补零，多于该值则截断；None 表示不调整通道数。

    Returns:
        Path: 已保存的 .h5 文件路径。
    
    Raises:
        FileExistsError: 输出文件存在且 overwrite=False 时抛出。
        Exception: 读取、转换或保存过程中出现的其他错误会向上抛出。
    """
    # 准备路径
    fif_path = Path(fif_file)
    h5_path = Path(h5_file)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    # 检查是否已存在
    if h5_path.exists() and not overwrite:
        raise FileExistsError(f"{h5_path} already exists. Use overwrite=True to replace.")

    print(f"Processing {fif_path.name} → {h5_path.name} ...", flush=True)

    # 读取 epochs，单位转换：V → μV
    epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
    X = (epochs.get_data() * 1e6).astype(np.float32)  # shape = (n_epochs, n_channels, n_times)

    # 调整通道数
    if target_channels is not None:
        n_epochs, n_channels, n_times = X.shape
        if n_channels < target_channels:
            print(f"Padding channels: {n_channels} → {target_channels}", flush=True)
            X_padded = np.zeros((n_epochs, target_channels, n_times), dtype=np.float32)
            X_padded[:, :n_channels, :] = X
            X = X_padded
        elif n_channels > target_channels:
            print(f"Truncating channels: {n_channels} → {target_channels}", flush=True)
            X = X[:, :target_channels, :]

    # 提取标签
    y = epochs.events[:, -1].astype(np.int32)

    # 保存到 HDF5
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('data', data=X, dtype='float32',compression="gzip",compression_opts=9)
        f.create_dataset('label', data=y, dtype='int32',compression="gzip",compression_opts=9)

    print(f"Saved to {h5_path}", flush=True)
    return h5_path














        
    