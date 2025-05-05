import os

def rename_files(directory_path):
    for filename in os.listdir(directory_path):
        # Kiểm tra định dạng file (số_idXX_idXX.mp4)
        if filename.count('_') == 2 and filename.endswith('.mp4'):
            parts = filename.split('_')
            
            if parts[0].isdigit() and parts[1].startswith('id') and parts[2].startswith('id'):
                # Lấy các phần ID
                id_part1 = parts[1]  # id62
                id_part2 = parts[2].split('.')[0]  # id67
                
                # Tạo tên mới
                new_name = f"{id_part1}_{id_part2}_0000.mp4"
                
                # Đổi tên file
                old_path = os.path.join(directory_path, filename)
                new_path = os.path.join(directory_path, new_name)
                os.rename(old_path, new_path)
                
                print(f"Đã đổi: {filename} → {new_name}")

# Sử dụng
directory_path = "E:\TLCN\Main\datasets\FMFCCV/fake_rename"
rename_files(directory_path)
