import os
import scipy.io
import torch
import pandas as pd


def read_data(mat_folder_path, excel_file_path, output_file_path):
    # 读取 Excel 文件获取 subjectID 和 group 信息
    excel_data = pd.read_excel(excel_file_path)
    sample_group_map = dict(zip(excel_data['Subject'], excel_data['Group']))

    # 初始化数据列表
    data_list = []
    id_list = []
    label_list = []

    # 遍历每个 .mat 文件
    for filename in os.listdir(mat_folder_path):
        if filename.endswith(".mat"):
            # 提取 subjectID
            parts = filename.split('.')[0].split('_')
            subject_id = '_'.join(parts[1:-1])  # 不包含序号的被试ID
            sample_id = '_'.join(parts[1:])  # 包含序号的完整样本ID

            file_path = os.path.join(mat_folder_path, filename)
            mat_data = scipy.io.loadmat(file_path)

            # 读取数据
            if 'ROISignals' in mat_data:
                # 获取标签
                if sample_id in sample_group_map:
                    group = sample_group_map[sample_id]
                    # if group == 'CN' or group == 'AD':
                    if group == 'CN' or group == 'EMCI' or group == 'LMCI':
                    # if group == 'AD' or group == 'EMCI' or group == 'LMCI':
                    # if group == 'EMCI' or group == 'LMCI':
                    # if group == 'CN' or group == 'EMCI':
                        data_tensor = torch.tensor(mat_data['ROISignals'])
                        # 前90个脑区
                        data_tensor = data_tensor[:, :90]
                        data_list.append(data_tensor)
                        id_list.append(subject_id)
                    # # 四分类
                    # if group == 'CN':
                    #     label = 0
                    # elif group == 'EMCI':
                    #     label = 1
                    # elif group == 'LMCI':
                    #     label = 2
                    # elif group == 'AD':
                    #     label = 3
                    # else:
                    #     raise ValueError(f"未知的类别: {group}")

                    # 两分类
                    if group == 'CN':
                        label = 0
                    elif group == 'EMCI' or group == 'LMCI':
                        label = 1
                    else:
                        continue
                    # if group == 'EMCI':
                    #     label = 0
                    # elif group == 'LMCI':
                    #     label = 1
                    # else:
                    #     continue

                    label_list.append(label)

    # 合并数据并保存
    all_data = torch.stack(data_list)
    all_labels = torch.tensor(label_list)
    all_ids = id_list
    torch.save({'all_data': all_data, 'all_labels': all_labels, 'all_ids': all_ids}, output_file_path)
    print("数据已成功保存到 .pt 文件中！")
