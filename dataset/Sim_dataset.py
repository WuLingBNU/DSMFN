import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.wavelet_transform import wavelet_transform, calculate_pearson_for_each_band
from collections import defaultdict
import gc


class DualDataSet(Dataset):
    def __init__(self, data, label, size, step, wavelet='db20', level=3, delete_nan=False):
        super(DualDataSet, self).__init__()
        assert len(data.shape) == 3
        data = data.unfold(-1, size=size, step=step).transpose(1, 2)  # data的形状是[被试，窗口，脑区，时间点]
        self.num_window = data.shape[1]
        if len(label.shape) > 1:
            label = label.squeeze(-1)

        # 归一化
        data_min, _ = data.min(2)
        data_max, _ = data.max(2)
        data = (data - data_min.unsqueeze(2)) / (data_max.unsqueeze(2) - data_min.unsqueeze(2))

        # 是否删除包含nan的样本，还是只将nan变成0
        if delete_nan:
            have_nan = torch.isnan(data).reshape(data.size(0), -1)
            have_nan = torch.any(have_nan, dim=1)
            data = data[~have_nan]
        else:
            data = torch.nan_to_num(data, nan=0.0)
        if torch.isnan(data).any():
            raise ValueError("have nan")

        self.data = data
        self.label = label
        self.wavelet = wavelet
        self.level = level
        self.cache = defaultdict(tuple)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        if item in self.cache:
            return self.cache[item]
        raw_data = self.data[item]

        wavelet_data, length_list = wavelet_transform(raw_data.transpose(-1, -2).numpy()
                                                      , wavelet=self.wavelet, level=self.level)

        person_matrix = calculate_pearson_for_each_band(wavelet_data, length_list, False)  # 【3，116，116，4】
        person_matrix[np.isnan(person_matrix)] = 0
        person_matrix = person_matrix.swapaxes(-1, 1)  # [3,4,116,116]

        if np.isnan(person_matrix).any():
            print(item)
            raise ValueError
        try:  # 双数据流：初始滑窗信号，经过滑窗后小波变换后各个频段的功能连接
            self.cache[item] = ((raw_data.to(torch.float32), person_matrix.astype(np.float32)), self.label[item])
        except MemoryError:
            self._evict_cache()
            # 尝试再次存储（此时内存应该已经释放）
            self.cache[item] = ((raw_data.to(torch.float32), person_matrix.astype(np.float32)), self.label[item])

        return self.cache[item]

    def _evict_cache(self):
        """删除缓存中的某个项目以释放空间"""
        if self.cache:
            # 随便删除缓存中的一个项目（可以换成更复杂的策略）
            evict_key = next(iter(self.cache.keys()))
            print(f"Evicting cached item with index: {evict_key}")
            del self.cache[evict_key]
            gc.collect()


def get_dual_data_loader(x: torch.Tensor, y: torch.Tensor, wavelet, delete_nan, window_size: int, window_step: int
                         , batch_size: int = 16, shuffle=True, level: int = 3, num_worker: int = 3):
    persistent = True if num_worker >= 1 else False
    m_set = DualDataSet(x, y, window_size, window_step, level=level, wavelet=wavelet, delete_nan=delete_nan)
    loader = DataLoader(dataset=m_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker,
                        persistent_workers=persistent)

    return loader
