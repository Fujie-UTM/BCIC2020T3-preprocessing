# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 16:06:40 2025

@author: Fujie
"""

import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
import seaborn as sns
import pandas as pd
from typing import Callable, List, Optional, Sequence, Tuple, Union
#%%

def cm_to_inch(value: float) -> float:
    """厘米转英寸"""
    return value / 2.54


def set_global_plot_style(
    width_cm: float = 8,
    height_cm: float = 6,
    file_type: str ="png",
    font_family: str = "Times New Roman",
    font_size: int = 5,
    dpi: int = 600,
):
    """设置 Matplotlib 全局绘图风格（适合无损保存 tiff/svg 等）

    Args:
        width_cm (float): 图形宽度（厘米）
        height_cm (float): 图形高度（厘米）
        font_family (str): 全局字体
        font_size (int): 全局字体大小
        dpi (int): 图像分辨率
    """
    plt.rcParams['figure.figsize'] = (cm_to_inch(width_cm), cm_to_inch(height_cm))
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['figure.constrained_layout.use'] = False
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 6
    plt.rcParams['figure.titlesize'] = 0
    plt.rcParams['legend.title_fontsize'] = 0
    plt.rcParams['legend.fontsize'] = 6
    plt.rcParams['legend.markerscale'] = 0.8
    plt.rcParams['legend.columnspacing'] = 0.5
    plt.rcParams['legend.borderaxespad'] = 0.5
    plt.rcParams['legend.borderpad'] = 0
    plt.rcParams['legend.framealpha'] = 0
    plt.rcParams['legend.labelspacing'] = 0.1
    plt.rcParams['legend.handlelength'] = 1.0
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['savefig.format'] = file_type
    plt.rcParams['savefig.bbox'] = 'standard'
    
    
def plot_IC_bandwise_MI(
    df: pd.DataFrame,
    timewin_name_list: List[str],
    IC_types_list: List[str],
    BG_name_list: List[str],
    save_dir: Union[str, Path],
    skip_empty: bool = True
) -> None:
    """批量绘制 MI boxplot 并保存图像。

    Args:
        df: 包含 'time'、'band' 以及各 IC 类型列的 DataFrame。
        timewin_name_list: 时间窗名称列表。
        IC_types_list: 需要绘制的 IC 类型名称列表（列名）。
        BG_name_list: 频带顺序列表（用于 boxplot 的 order 参数）。
        save_dir: 保存图片的目录。
        skip_empty: 若某个过滤条件下无数据，是否跳过绘图。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 基本列检查
    required_cols = {'time', 'band'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame 缺少必要列：{sorted(missing)}")

    for timewin_name, IC_type in product(timewin_name_list, IC_types_list):
        if IC_type not in df.columns:
            print(f"[跳过] 列不存在：{IC_type}")
            continue

        df_filtered = df[df['time'] == timewin_name]
        if df_filtered.empty and skip_empty:
            print(f"[跳过] time={timewin_name} 数据为空")
            continue

        fig, ax = plt.subplots()

        sns.boxplot( data=df_filtered, x='band', y=IC_type, order=BG_name_list,
            color='lavender', saturation=1, dodge=True, width=0.5, gap=0,                 # 若 seaborn 版本不支持可去掉
            whis=1.5, linecolor='black', linewidth=0.5, fliersize=0.5,
            flierprops={"marker": "x"}, hue_norm=None, native_scale=False,
            log_scale=None, formatter=None, legend='brief', ax=ax
        )

        ax.set_xlabel("Frequency Band")
        ax.set_ylabel("MI (bit)")

        fig.savefig(save_dir / f"{timewin_name}_{IC_type}_MI")
        plt.close(fig)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        