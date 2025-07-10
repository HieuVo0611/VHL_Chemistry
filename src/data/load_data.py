import os
import pandas as pd
import re
from typing import Literal
from src.config import *

def load_xlsx_file(file_path):
    df = pd.read_excel(file_path)
    return df

def swp2df(swp_path: str) -> pd.DataFrame:
    """
    Load a SWP file and convert it to a DataFrame.
    """
    all_data = pd.DataFrame()
    # Duyệt qua từng file trong thư mục
    for filename in os.listdir(swp_path):
        if filename.endswith(".swp"):
            file_path = os.path.join(swp_path, filename)

            # Đọc file .swp
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # Xử lý dữ liệu số liệu
            vertical_axis = ''
            horizontal_axis = ''
            data_start = False
            data = []

            for line in lines:
                if 'Vertical Axis' in line:
                    vertical_axis = line.split('\t')[1].strip()
                if 'Horizonal Axis' in line:
                    horizontal_axis = line.split('\t')[1].strip()
                    data_start = True
                    continue
                if data_start and line.strip() == '':
                    continue
                if data_start:
                    values = line.strip().split('\t')
                    if len(values) == 2:
                        data.append([float(values[0]), float(values[1])])

            # Tạo dataframe cho file hiện tại
            df = pd.DataFrame(data, columns=[horizontal_axis, vertical_axis])

            # Đổi tên cột thành 'j(mA)' và 'U(V)'
            df.rename(columns={horizontal_axis: 'X', vertical_axis:'Y'}, inplace=True)

            # Thêm cột 'Filenames' với giá trị là tên file
            label_file = filename.strip(".swp")
            df['Filenames'] = label_file

            # Tạo cột 'label' với nội dung số nằm giữa "Pb" và "L1" hoặc "L2"
            pb_match = re.search(r"Pb\s*([\d\.]+)", label_file)
            cd_match = re.search(r"Cd\s*([\d\.]+)", label_file)
            if pb_match and cd_match:
                pb_label = pb_match.group(1)
                cd_label = cd_match.group(1)
            else:
                pb_label, cd_label = "",""  # Trường hợp không tìm thấy số
            df['Pb'] = pb_label
            df['Cd'] = cd_label

            # Append dataframe hiện tại vào all_data
            all_data = pd.concat([all_data, df[['Filenames','X','Y', 'Pb', 'Cd']]], ignore_index=True)

    return all_data

def swp2df_V250710(swp_path: str) -> pd.DataFrame:
    """
    Load tất cả file .swp trong thư mục swp_path, trả về DataFrame chứa các cột: Filenames, X, Y, Pb, Cd.
    """
    all_data = pd.DataFrame()
    for filename in os.listdir(swp_path):
        if filename.endswith(".swp"):
            file_path = os.path.join(swp_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            vertical_axis = ''
            horizontal_axis = ''
            data_start = False
            data = []

            for line in lines:
                if 'Vertical Axis' in line:
                    vertical_axis = line.split('\t')[1].strip()
                if 'Horizonal Axis' in line:
                    horizontal_axis = line.split('\t')[1].strip()
                    data_start = True
                    continue
                if data_start and line.strip() == '':
                    continue
                if data_start:
                    values = line.strip().split('\t')
                    if len(values) == 2:
                        try:
                            data.append([float(values[0]), float(values[1])])
                        except Exception:
                            continue

            df = pd.DataFrame(data, columns=[horizontal_axis, vertical_axis])
            df.rename(columns={horizontal_axis: 'X', vertical_axis: 'Y'}, inplace=True)
            label_file = filename.strip(".swp")
            df['Filenames'] = label_file

            # Lấy nhãn Pb, Cd từ tên file, ví dụ: Pb0.1_Cd0 L1.swp
            pb_match = re.search(r"Pb\s*([\d\.]+)", label_file)
            cd_match = re.search(r"Cd\s*([\d\.]+)", label_file)
            if pb_match and cd_match:
                pb_label = pb_match.group(1)
                cd_label = cd_match.group(1)
            else:
                pb_label, cd_label = "", ""
            df['Pb'] = pb_label
            df['Cd'] = cd_label

            all_data = pd.concat([all_data, df[['Filenames', 'X', 'Y', 'Pb', 'Cd']]], ignore_index=True)
    return all_data

def extract_labels_from_filename(filename, label):
    
    # Ví dụ: "3. ENR 0.3uM - CIP 0.5uM.xlsx"
    enr_match = re.search(r'ENR\s([\d\.]+)uM', filename)
    cip_match = re.search(r'CIP\s([\d\.]+)uM', filename)
    if enr_match and cip_match:
        enr = float(enr_match.group(1))
        cip = float(cip_match.group(1))
        return enr, cip
    else:
        raise ValueError(f"Filename {filename} không đúng format label")

    

    

def load_dataset(folder_path, label:Literal['ENR','PB'] = 'PB'):
    
    if label == 'ENR':
        data_list = []
        labels_list = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.xlsx'):
                file_path = os.path.join(folder_path, filename)
                df = load_xlsx_file(file_path)

                # Extract label
                enr, cip = extract_labels_from_filename(filename, label)

                data_list.append(df)
                labels_list.append([enr, cip])

        return data_list, labels_list
    
    elif label == 'PB':
        data_list = []
        labels_list = []
        pb_cd_df = swp2df(folder_path)
        pb_cd_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'metadata_pb_cd.csv'), index=False)
        for sample, group in pb_cd_df.groupby('Filenames'):
            # Chọn cột 'U(V)' và 'J(mA)' cho mỗi mẫu
            sample_df = group[['X', 'Y']]
            data_list.append(sample_df)

            # Lấy nhãn Pb và Cd từ cột tương ứng
            pb_label = group['Pb'].iloc[0]
            cd_label = group['Cd'].iloc[0]
            labels_list.append([pb_label, cd_label])
        return data_list, labels_list

def load_dataset_V250710(folder_path):
    """
    Load dataset từ folder chứa các file .swp (dùng cho data mới HP4), trả về data_list và labels_list.
    """
    data_list = []
    labels_list = []
    pb_cd_df = swp2df_V250710(folder_path)
    # Có thể lưu lại metadata nếu muốn
    pb_cd_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'metadata_pb_cd_V250710.csv'), index=False)
    for sample, group in pb_cd_df.groupby('Filenames'):
        sample_df = group[['X', 'Y']]
        data_list.append(sample_df)
        pb_label = group['Pb'].iloc[0]
        cd_label = group['Cd'].iloc[0]
        labels_list.append([pb_label, cd_label])
    return data_list, labels_list

if __name__ == "__main__":
    a = swp2df('data\\raw\pb_cd')
    print(a)