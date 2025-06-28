import os
import scipy.io
import torch
import pandas as pd


def read_data(mat_folder_path, excel_file_path, output_file_path):
    excel_data = pd.read_excel(excel_file_path)
    sample_group_map = dict(zip(excel_data['Subject'], excel_data['Group']))

    data_list = []
    id_list = []
    label_list = []

    for filename in os.listdir(mat_folder_path):
        if filename.endswith(".mat"):
            parts = filename.split('.')[0].split('_')
            subject_id = '_'.join(parts[1:-1])   
            sample_id = '_'.join(parts[1:])  

            file_path = os.path.join(mat_folder_path, filename)
            mat_data = scipy.io.loadmat(file_path)

            if 'ROISignals' in mat_data:
                if sample_id in sample_group_map:
                    group = sample_group_map[sample_id]
                    if group == 'CN' or group == 'EMCI' or group == 'LMCI':
                        data_tensor = torch.tensor(mat_data['ROISignals'])
                        data_tensor = data_tensor[:, :90]
                        data_list.append(data_tensor)
                        id_list.append(subject_id)
                    
                    if group == 'CN':
                        label = 0
                    elif group == 'EMCI' or group == 'LMCI':
                        label = 1
                    else:
                        continue

                    label_list.append(label)

    all_data = torch.stack(data_list)
    all_labels = torch.tensor(label_list)
    all_ids = id_list
    torch.save({'all_data': all_data, 'all_labels': all_labels, 'all_ids': all_ids}, output_file_path)
