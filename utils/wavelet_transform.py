import pywt
import numpy as np
import torch
from typing import List


def wavelet_transform(time_series: np.ndarray, wavelet='db20', level=3):
    """
    对输入时间序列数据执行小波分解。

    参数:
    - time_series (list of numpy arrays): 输入的时间序列数据，形状为 [序列个数, 数据点, 时间步]
    - wavelet (str): 使用的小波类型，默认 'db20'
    - level (int): 小波分解的层级，默认 3

    返回:
    - waveltsDatas (list of numpy arrays): 分解后的小波系数数据列表，形状为 [序列个数, 系数数, 数据点]
    """
    waveltsDatas = []
    len_list = []
    for i in range(len(time_series)):
        waveletData = []
        data = time_series[i].T  # 转置成 [时间步, 数据点] 的形状

        # 对每一个数据点进行小波分解
        for r in range(data.shape[0]):  # 遍历每一行
            coeffs = pywt.wavedec(data[r], wavelet, level=level)
            len_list = [len(i) for i in coeffs]
            concatenated_coeffs = np.concatenate(coeffs)  # 拼接系数
            waveletData.append(concatenated_coeffs)

        # 转换为数组并存储分解结果
        waveletData = np.array(waveletData)
        waveltsDatas.append(waveletData.T)  # 转置以返回原始顺序
    np_wavelet_data = np.stack(waveltsDatas, axis=0)
    return np_wavelet_data, len_list


def calculate_pearson_for_each_band(wavelet_data: np.ndarray, length_list: List, return_tensor=True):
    correlated = calculate_pearson(wavelet_data, length_list)
    return torch.from_numpy(correlated) if return_tensor else correlated
