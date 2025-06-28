import pywt
import numpy as np
import torch
from typing import List


def wavelet_transform(time_series: np.ndarray, wavelet='db20', level=3):
    waveltsDatas = []
    len_list = []
    for i in range(len(time_series)):
        waveletData = []
        data = time_series[i].T  

        for r in range(data.shape[0]):  
            coeffs = pywt.wavedec(data[r], wavelet, level=level)
            len_list = [len(i) for i in coeffs]
            concatenated_coeffs = np.concatenate(coeffs)  
            waveletData.append(concatenated_coeffs)

        waveletData = np.array(waveletData)
        waveltsDatas.append(waveletData.T)  
    np_wavelet_data = np.stack(waveltsDatas, axis=0)
    return np_wavelet_data, len_list


def calculate_pearson_for_each_band(wavelet_data: np.ndarray, length_list: List, return_tensor=True):
    correlated = calculate_pearson(wavelet_data, length_list)
    return torch.from_numpy(correlated) if return_tensor else correlated
