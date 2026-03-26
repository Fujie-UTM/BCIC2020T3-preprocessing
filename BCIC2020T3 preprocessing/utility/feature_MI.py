# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 15:00:12 2025

@author: Fujie
"""
"""IC 互信息（Mutual Information）分析与汇总模块。

本模块提供以下功能（遵循 Google Python Style Guide 的 docstring 规范，中文撰写）：
1. 计算每个独立成分（Independent Component, IC）与离散标签之间的互信息并导出结果（`cal_IC_MI`）。
2. 从多个目录读取同名 MNE Epochs 并拼接（`concat_epochs_from_folderlist`）。
3. 在给定特征提取器的情况下，先做特征提取，再计算 IC 的互信息（`cal_one_feature_extraction_MI`）。
4. 从互信息结果 CSV 中按 IC 类型做加总（仅白名单类型），得到单行摘要（`summarize_IC_MI`）。
5. 批量汇总多个被试 / 时间窗 / 频段的互信息结果，生成明细与聚合表（`build_mi_summaries`）。

使用注意：
- 互信息采用 sklearn 的 `mutual_info_classif`，固定 `n_neighbors=3`。
- 输入标签 y 会强制转为 `int`，若不能安全转换将抛出异常。
- 对数据可选做 z-score 标准化（默认开启）。
"""

# 标准库
from itertools import product
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union
import copy

# 第三方
import mne
import numpy as np
import pandas as pd
import math
from sklearn.base import BaseEstimator
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

#%%
#%%
IC_TYPES = ['brain', 'muscle artifact', 'eye blink', 'heart beat',
            'line noise', 'channel noise', 'other',]  # 定义白名单 IC 类型顺序


def cal_IC_MI(
    data: np.ndarray,  # 输入特征矩阵，形状 (n_samples, n_ICs)
    labels: np.ndarray,  # 离散标签向量，形状 (n_samples,)
    ic_check_df: pd.DataFrame,  # 含 IC 元信息的 DataFrame，至少包含列 ['IC_name','Type','Proba']
    ic_names: List[str],  # IC 名称列表，长度应等于 data 的列数
    random_state: Optional[int] = None,  # 随机种子，透传给 mutual_info_classif
    n_jobs: Optional[int] = None,   # 透传给 mutual_info_classif
    csv_path: Optional[str] = "IC_MI.csv",  # 可选：导出 CSV 的路径；为 None/空则不导出
    is_zscore: bool = True,  # 是否对 X 做 z-score 标准化
) -> pd.DataFrame:
    """
    计算每个 IC 相对于离散标签的互信息并汇总输出。

    功能概述：
        - 固定使用 kNN 估计互信息，n_neighbors=3（不做小样本自适应）。
        - 将 y 强制转换为 int（无法安全转换将报错）。
        - 可选对 X 做 z-score（默认开启）。
        - 连接 ic_check_df 元数据（Type, Proba）并输出排名、占比及累计指标。
        - 可选将结果导出为 CSV。

    Args:
        data (np.ndarray): 特征矩阵，形状为 (n_samples, n_ICs)。每列对应一个 IC。
        labels (np.ndarray): 离散标签，形状为 (n_samples,)。将被强制转换为整型。
        ic_check_df (pd.DataFrame): IC 标注表，需包含列 {'IC_name','Type','Proba'}。
        ic_names (List[str]): IC 名称列表，长度需等于 data 的列数。
        random_state (Optional[int], optional): 随机种子，传入 sklearn 的 MI 函数。默认为 None。
        n_jobs (Optional[int], optional): 并行作业数（sklearn 参数）。默认为 None（单线程）。
        csv_path (Optional[str], optional): 导出结果 CSV 的路径；为 None 或空字符串则不导出。默认为 "IC_MI.csv"。
        is_zscore (bool, optional): 是否对 X 做标准化（均值 0 方差 1）。默认为 True。

    Returns:
        pd.DataFrame: 包含以下列的 DataFrame（按 Mutual_info 降序）：
            - 'IC_name': IC 名称
            - 'Mutual_info': 互信息值
            - 'Rank': 互信息排名（稠密排名，1 表示最大）
            - 'Proportion': 当前 IC 的 MI 占总 MI 的比例
            - 'Type': 来自 ic_check_df 的类型
            - 'Proba': 来自 ic_check_df 的概率（或置信度）
            - 'Cum_Mutual_info': 按排序累计 MI
            - 'Cum_Proportion': 按排序累计占比

    Raises:
        ValueError: 当输入维度不符、含 NaN/Inf、类别数不足、ic_check_df 缺列/重复/缺失映射、
                    或样本数不满足 kNN 互信息估计需求等情况时抛出。
        FileNotFoundError: 当需要导出 CSV 且路径父目录创建失败（极端情况）时可能引发（由底层抛出）。
    """

    # —— 基础校验 —— #
    if data.ndim != 2:  # 校验 data 维度为二维
        raise ValueError("`data` must be 2D array of shape (n_samples, n_ICs).")
    if labels.ndim != 1:  # 校验 labels 维度为一维
        raise ValueError("`labels` must be 1D array of shape (n_samples,).")
    if data.shape[0] != labels.shape[0]:  # 样本数需一致
        raise ValueError("`data` rows must match length of `labels`.")
    if len(ic_names) != data.shape[1]:  # 列数与 ic_names 长度需一致
        raise ValueError("`ic_names` length must equal number of ICs (data.shape[1]).")

    # 转换/压平
    X = np.asarray(data, dtype=float)  # 将输入转换为 float 的 ndarray
    y = np.asarray(labels, dtype=int).ravel()  # 将标签转为 int，并压平为一维

    # 缺失/非常数检查
    if not np.isfinite(X).all():  # 检查 X 中是否存在 NaN/Inf
        raise ValueError("`data` contains NaN or inf; please clean your data before MI.")
    if not np.isfinite(y).all():  # 检查 y 中是否存在 NaN/Inf
        raise ValueError("`labels` contains NaN or inf.")

    # 类别数
    n_classes = np.unique(y).size  # 计算离散标签的唯一类别数
    if n_classes < 2:  # 至少需要两类才能计算互信息
        raise ValueError("`labels` must contain at least 2 classes for mutual information.")

    # —— IC 元信息：列存在性 + 对齐 + 冲突提示 —— #
    required_cols = {'IC_name', 'Type', 'Proba'}  # 需要的列集合
    if not required_cols.issubset(ic_check_df.columns):  # 检查缺失列
        missing_cols = required_cols - set(ic_check_df.columns)  # 计算缺失列集合
        raise ValueError(f"`ic_check_df` 缺少列: {sorted(missing_cols)}")

    dup_counts = ic_check_df['IC_name'].value_counts()  # 统计 IC_name 是否有重复
    if (dup_counts > 1).any():  # 若存在重复的 IC_name
        dup_names = dup_counts[dup_counts > 1].index.tolist()[:5]  # 展示前 5 个重复名
        raise ValueError(f"`ic_check_df` 中存在重复 IC_name：{dup_names} ... (仅示例前5个)")

    ic_meta = ic_check_df[['IC_name', 'Type', 'Proba']].set_index('IC_name')  # 取必要列并以 IC_name 索引
    missing = set(ic_names) - set(ic_meta.index)  # 找出在标注表中缺失的 IC 名称
    if missing:  # 若存在缺失映射
        head = sorted(missing)[:5]  # 展示前 5 个
        raise ValueError(f"IC 名称在标注表中缺失：{head} ... (共 {len(missing)} 个)")

    # —— （可选）标准化 —— #
    if is_zscore:  # 若启用 z-score
        X = StandardScaler().fit_transform(X)  # 对每列（IC）做标准化

    # —— 互信息 —— #
    n_neighbors = 3  # 固定近邻数为 3
    if X.shape[0] <= n_neighbors:  # 样本数需大于 n_neighbors
        raise ValueError(
            f"n_samples ({X.shape[0]}) must be > n_neighbors ({n_neighbors}). "
            "请增加样本或调整 n_neighbors。"
        )
    mi_vals = mutual_info_classif(  # 计算每个特征（IC）与标签之间的互信息
        X=X,  # 特征矩阵
        y=y,  # 整型标签
        discrete_features=False,  # 特征视为连续
        n_neighbors=n_neighbors,  # 近邻数
        copy=True,  # 复制数据
        random_state=random_state,  # 随机种子
        n_jobs=n_jobs,   # 这里正式传给 sklearn
    )

    # —— 占比与排名 —— #
    total_mi = float(np.sum(mi_vals))  # 计算总互信息
    proportions = (mi_vals / total_mi) if total_mi > 0 else np.zeros_like(mi_vals, dtype=float)  # 占比
    ranks = pd.Series(mi_vals).rank(method="dense", ascending=False).astype(int).to_numpy()  # 稠密排名

    # —— 汇总输出 —— #
    out_df = pd.DataFrame({  # 构建结果 DataFrame
        'IC_name':     ic_names,  # IC 名称
        'Mutual_info': mi_vals,  # 互信息值
        'Rank':        ranks,  # 排名
        'Proportion':  proportions,  # 占比
    }).join(ic_meta, on='IC_name')  # 连接类型与概率元信息

    # 按 MI 降序并加累计
    out_df = out_df.sort_values('Mutual_info', ascending=False, kind='mergesort').reset_index(drop=True)  # 稳定排序
    out_df['Cum_Mutual_info'] = out_df['Mutual_info'].cumsum()  # 累计 MI
    out_df['Cum_Proportion']  = out_df['Proportion'].cumsum()  # 累计占比

    # —— 导出 —— #
    if csv_path:  # 若给定导出路径
        p = Path(csv_path)  # 构造路径对象
        if p.parent and not p.parent.exists():  # 若父目录不存在则递归创建
            p.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(p, index=False, encoding="utf-8")  # 保存为 UTF-8 编码 CSV

    return out_df  # 返回结果 DataFrame



def concat_epochs_from_folderlist(
    folder_list: Sequence[Union[str, Path]],  # 目录列表，每个目录下包含同名 epochs 文件
    filename: Union[str, Path],  # 需要在各目录内读取的相同文件名
) -> mne.Epochs:
    """
    从多个目录中读取同名 epochs 文件并拼接。

    Args:
        folder_list (Sequence[Union[str, Path]]): 目录路径序列，每个目录应包含同名 epochs 文件。
        filename (Union[str, Path]): 需要在各目录中读取的 epochs 文件名（相同）。

    Returns:
        mne.Epochs: 将各目录中读取的 Epochs 以 `mne.concatenate_epochs` 拼接后的结果。

    Raises:
        FileNotFoundError: 任一目录中找不到目标文件时抛出。
        ValueError: 若未成功读取任何 Epochs（列表为空）时抛出。
        RuntimeError: 当拼接时信息不一致且 `on_mismatch='raise'` 触发底层错误时可能抛出。
    """
    epochs_list = []  # 用于存放读取的 Epochs 对象

    for folder in folder_list:  # 遍历每个目录
        p = Path(folder) / filename  # 构造完整文件路径
        if not p.exists():  # 文件不存在则报错
            raise FileNotFoundError(f"找不到文件：{p}")
        # 视需求决定 preload（拼接时通常 preload=True 更稳）
        epochs = mne.read_epochs(p, preload=True, verbose=False)  # 读取 epochs，预加载数据
        epochs_list.append(epochs)  # 追加到列表

    if not epochs_list:  # 如果没有成功读取任何 Epochs
        raise ValueError("folder_list 为空或未成功读取任何 Epochs。")

    epochs_all = mne.concatenate_epochs(  # 拼接所有 Epochs
        epochs_list,
        add_offset=True,         # 不同录制间事件时间轴错位时更安全
        on_mismatch='raise',     # 信息不一致时直接报错（也可用 'ignore'）
        verbose=False,
    )
    return epochs_all  # 返回拼接结果



def cal_one_feature_extraction_MI(
    IC_folder_list: Sequence[Union[str, Path]],  # 各被试/会话的 IC Epochs 所在目录列表
    IC_data_name: Union[str, Path],  # 需要读取的 Epochs 文件名
    IC_csv_path: Union[str, Path],  # IC 标注表 CSV 路径
    output_csv_path: Union[str, Path],  # 输出互信息 CSV 路径
    feature_extractor: BaseEstimator,  # 特征提取器（应符合 sklearn 接口）
    random_state: Optional[int] = None,  # 随机种子
    n_jobs: Optional[int] = None,  # 并行作业数
    is_zscore: bool = True,  # 是否 z-score
):
    """
    读取多个目录下的同名 MNE Epochs 并拼接，使用给定特征提取器对 (n_epochs, n_ICs, n_times) 做特征提取，
    随后计算每个 IC 的互信息并导出 CSV。

    假设：
        - 特征提取器支持 3D 输入 (n_epochs, n_ICs, n_times) 并返回二维 (n_epochs, n_ICs)，
          即为每个 IC 聚合成 1 个数值特征。

    Args:
        IC_folder_list (Sequence[Union[str, Path]]): 存放 Epochs 的多个目录。
        IC_data_name (Union[str, Path]): 需要在每个目录内读取的同名 Epochs 文件名。
        IC_csv_path (Union[str, Path]): 包含 IC 元信息（IC_name, Type, Proba）的 CSV。
        output_csv_path (Union[str, Path]): 互信息结果 CSV 的输出路径。
        feature_extractor (BaseEstimator): sklearn 风格的特征提取器，需实现 `fit_transform`。
        random_state (Optional[int], optional): 随机种子，传入互信息函数。默认为 None。
        n_jobs (Optional[int], optional): 并行作业数，传入互信息函数。默认为 None。
        is_zscore (bool, optional): 是否对特征做 z-score 标准化。默认为 True。

    Returns:
        pd.DataFrame: `cal_IC_MI` 的返回结果 DataFrame。

    Raises:
        FileNotFoundError: 任何一个目录缺失目标文件时抛出（由 `concat_epochs_from_folderlist` 抛出）。
        ValueError: 特征维度与 IC 名称/标签长度不匹配，或输出不是 2D 等异常。
        RuntimeError: MNE 在拼接或读取时可能抛出的底层错误。
    """
    # 1) 读取并拼接 Epochs
    epochs_all = concat_epochs_from_folderlist(  # 调用上游函数，读取并拼接 Epochs
        folder_list=IC_folder_list,
        filename=IC_data_name,
    )

    ic_names = epochs_all.ch_names  # 假定这些就是 IC 名
    X_raw = epochs_all.get_data(copy=False)  # (n_epochs, n_ICs, n_times)
    y = epochs_all.events[:, -1]            # (n_epochs,)
    ic_check_df = pd.read_csv(IC_csv_path)  # 读取 IC 标注表

    # 2) 特征提取（假定提取器支持 3D 输入并返回每 IC 1 个特征）
    X_feat = feature_extractor.fit_transform(X_raw)  # 期望: (n_epochs, n_ICs)

    # --- 一致性检查 ---
    if X_feat.ndim != 2:  # 输出必须为二维
        raise ValueError(f"期望 X_feat 为 2D，得到 {X_feat.shape}")
    n_samples, n_ics_out = X_feat.shape  # 读取形状
    if n_samples != len(y):  # 样本数需与标签长度一致
        raise ValueError(f"样本数不一致: X_feat={n_samples}, y={len(y)}")
    if n_ics_out != len(ic_names):  # 每个 IC 应该输出 1 个聚合特征
        raise ValueError(
            "期望每个 IC 只输出 1 个特征："
            f"X_feat.shape[1]={n_ics_out}, 但 ic_names={len(ic_names)}。"
            "若你的提取器为每个 IC 输出多个特征，请先在特征维上做聚合或重命名并调整下游逻辑。"
        )

    # 3) 计算 MI（在 cal_IC_MI 内部会 z-score 与把 y 转为 int）
    mi_df = cal_IC_MI(  # 调用互信息计算函数
        data=X_feat,
        labels=y,
        ic_check_df=ic_check_df,
        ic_names=ic_names,
        random_state=random_state,
        n_jobs=n_jobs,
        csv_path=output_csv_path,
        is_zscore=is_zscore,
    )
    return mi_df  # 返回互信息结果


def summarize_IC_MI(csv_path: Optional[Union[str, Path]]) -> pd.DataFrame:
    """
    从含有 `Type` 与 `Mutual_info` 列的 CSV 计算各“已知类别”的互信息之和，并返回单行 DataFrame。

    规则：
        - 仅统计白名单中的类别：IC_TYPES = ['brain', 'muscle artifact', 'eye blink',
          'heart beat', 'line noise', 'channel noise', 'other', 'total']。
        - 未在白名单的类型将被丢弃且不计入 total。
        - 缺失的已知类别补 0。
        - 输出列顺序固定为各已知类别 + 'total'。

    Args:
        csv_path (Optional[Union[str, Path]]): 输入 CSV 路径；文件需至少包含列 ['Type', 'Mutual_info']。

    Returns:
        pd.DataFrame: 单行 DataFrame，列顺序为 IC_TYPES + ['total']。

    Raises:
        ValueError: 当 csv_path 为空时抛出。
        FileNotFoundError: 当指定的 CSV 文件不存在时抛出。
        ValueError: 当 CSV 中缺少必须列时（由 pandas 读取 usecols 的错误或后续逻辑引发）。
    """
    if not csv_path:  # 参数为空校验
        raise ValueError("csv_path 不能为空。")
    p = Path(csv_path)  # 路径对象
    if not p.exists():  # 文件存在性校验
        raise FileNotFoundError(f"找不到文件：{p}")

    # 只读必要列
    df = pd.read_csv(p, usecols=["Type", "Mutual_info"])  # 仅加载必要列以提高健壮性与效率

    # 轻度清洗：去除首尾空格（保留大小写，严格匹配）
    df["Type"] = df["Type"].astype(str).str.strip()  # 去除类型首尾空格
    # 数值化 MI，无法解析的按 0 处理（也可以改为 dropna）
    df["Mutual_info"] = pd.to_numeric(df["Mutual_info"], errors="coerce").fillna(0.0)  # 将非数值转为 NaN 再填 0

    # 仅保留白名单类别（严格区分大小写）
    df = df[df["Type"].isin(IC_TYPES)]  # 过滤到白名单类型

    # 分组求和并对齐所有已知类别
    sums = (
        df.groupby("Type", observed=True)["Mutual_info"]  # 按 Type 分组求和
          .sum()
          .reindex(IC_TYPES, fill_value=0.0)  # 以白名单顺序对齐，缺失补 0
    )

    # 单行输出 + total（仅为白名单类别之和）
    result = sums.to_frame().T  # 转为单行 DataFrame
    result["total"] = float(sums.sum())  # 计算 total，为白名单之和
    result = result[IC_TYPES + ["total"]]  # 固定列顺序
    return result  # 返回摘要结果



def build_mi_summaries(
    timewin_name_list: Sequence[str],  # 时间窗名称列表
    BG_name_list: Sequence[str],  # 频段（band group）名称列表
    sub_list: Sequence[int],  # 被试编号列表
    mi_csv_dir: Union[str, Path],  # 存放（也输出）CSV 的目录
    filename_pattern: str = "sub{sub}_{timewin}_{band}_IC_MI.csv",  # 输入文件名模板
    agg: Union[str, Callable] = "median",  # 默认字符串，但可传函数
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    汇总多个被试、多时间窗、多频段的 IC-MI 结果，并生成明细表和聚合表。

    处理流程：
        1. 根据三元组 (timewin, band, sub) 构造输入 CSV 路径；
        2. 调用 `summarize_IC_MI` 读入并得到单行摘要；
        3. 逐行合并为“明细表 results”；
        4. 对明细表按 ['time','band'] 分组，对 IC 类型列使用 `agg` 聚合，得到“聚合表 mi_tf_sum”；
        5. 将两张表分别保存为 'MI_total.csv' 与 'MI_tf_sum.csv'。

    Args:
        timewin_name_list (Sequence[str]): 时间窗名称列表。
        BG_name_list (Sequence[str]): 频段名称列表（如 delta/theta/alpha 等）。
        sub_list (Sequence[int]): 被试编号列表。
        mi_csv_dir (Union[str, Path]): 互信息 CSV 所在目录（若不存在将创建）。
        filename_pattern (str, optional): 输入文件名模板，需包含占位 {sub},{timewin},{band}。默认为
            "sub{sub}_{timewin}_{band}_IC_MI.csv"。
        agg (Union[str, Callable], optional): 聚合方式，既可为 pandas 支持的字符串（如 'mean','median'），
            也可为可调用函数。默认为 "median"。

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - results: 明细表，每条 (sub, time, band) 一行，包含各 IC 类型互信息之和与 total。
            - mi_tf_sum: 聚合表，索引为 (time, band) 的聚合结果。

    Raises:
        FileNotFoundError: 当某个应存在的输入 CSV 文件不存在时抛出。
        ValueError: 当所有输入均为空导致无法构建结果表时（虽代码已有防护）或 `summarize_IC_MI` 内部异常等。
        OSError: 当写入 CSV 文件失败时可能由底层抛出。
    """
    mi_csv_dir = Path(mi_csv_dir)  # 规范化目录路径
    mi_csv_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在

    rows = []  # 用于累积每个 (sub,time,band) 的单行摘要
    IC_types_all = IC_TYPES + ['total']  # 

    for timewin_name, BG_name, sub_i in product(timewin_name_list, BG_name_list, sub_list):  # 三重笛卡尔积遍历
        filename = filename_pattern.format(sub=sub_i, timewin=timewin_name, band=BG_name)  # 构造文件名
        input_csv_path = mi_csv_dir / filename  # 完整路径

        if not input_csv_path.exists():  # 若文件不存在，直接报错（严格）
            raise FileNotFoundError(f"找不到文件：{input_csv_path}")

        sub_mi_df = summarize_IC_MI(csv_path=input_csv_path)  # 预期为单行 DataFrame
        # 补全缺失列
        for col in IC_types_all:  # 逐个检查/补齐列（保持原逻辑与写法）
            if col not in sub_mi_df.columns:
                sub_mi_df[col] = 0  # 缺失则补 0

        sub_mi_df.insert(0, 'sub',  sub_i)  # 加入被试编号列
        sub_mi_df.insert(1, 'time', timewin_name)  # 加入时间窗列
        sub_mi_df.insert(2, 'band', BG_name)  # 加入频段列
        rows.append(sub_mi_df)  # 收集行

    if not rows:  # 若没有任何行（极端情况）
        results = pd.DataFrame(columns=IC_types_all + ['sub', 'time', 'band'])  # 构造空明细表
        mi_tf_sum = pd.DataFrame(columns=['time', 'band'] + IC_types_all)  # 构造空聚合表
        return results, mi_tf_sum  # 返回空表

    # 合并明细
    results = pd.concat(rows, ignore_index=True)  # 合并为一张明细表
    results.to_csv(mi_csv_dir / 'MI_total.csv', index=False, encoding="utf-8")  # 导出明细

    # 聚合：用 agg 而不是 apply
    mi_tf_sum = (
        results.groupby(['time', 'band'], as_index=False)[IC_types_all]  # 对目标列做聚合
        .agg(agg)  # 使用给定聚合器（字符串或函数）
    )

    mi_tf_sum.to_csv(mi_csv_dir / 'MI_tf_sum.csv', index=False, encoding="utf-8")  # 导出聚合表

    return results, mi_tf_sum  # 返回（明细表, 聚合表）


def convert_mi_nat_to_bit(df: pd.DataFrame, 
                          ic_types: List[str] =IC_TYPES) -> pd.DataFrame:
    """将互信息由 nat 转换为 bit（除以 ln(2) ≈ 0.6931）。

    Args:
        df (pd.DataFrame): 包含互信息列的 DataFrame。
        ic_types (List[str]): 需要转换的列名列表（通常是 IC_TYPES）。

    Returns:
        pd.DataFrame: 转换后的 DataFrame（与输入对象相同）。
    """
    df = df.copy()
    for ic in ic_types:
        df.loc[:, ic] = df.loc[:, ic] / math.log(2)  # nat → bit
    return df


def convert_mi_bit_to_nat(df: pd.DataFrame, 
                          ic_types: List[str]=IC_TYPES) -> pd.DataFrame:
    """将互信息由 bit 转换为 nat（乘以 ln(2) ≈ 0.6931）。

    Args:
        df (pd.DataFrame): 包含互信息列的 DataFrame。
        ic_types (List[str]): 需要转换的列名列表（通常是 IC_TYPES）。

    Returns:
        pd.DataFrame: 转换后的 DataFrame（与输入对象相同）。
    """
    df = df.copy()
    for ic in ic_types:
        df.loc[:, ic] = df.loc[:, ic] * math.log(2)  # bit → nat
    return df


























