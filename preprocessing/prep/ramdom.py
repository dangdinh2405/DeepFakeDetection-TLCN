import os
import random

def random_files_to_txt(input_folder, output_file):
    # Lấy danh sách tất cả các file trong thư mục
    all_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    if not all_files:
        print("Thư mục không có file nào.")
        return
    
    # Tính 20% số lượng file
    num_files_to_select = max(1, int(len(all_files) * 0.2))  # Đảm bảo chọn ít nhất 1 file
    
    # Chọn ngẫu nhiên các file
    selected_files = random.sample(all_files, num_files_to_select)
    
    # Ghi vào file txt
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_name in selected_files:
            f.write('1 Celeb-real/'+ file_name + '\n')
    
    print(f"Đã ghi {num_files_to_select} file ngẫu nhiên vào {output_file}")

# Sử dụng hàm
input_folder = "E:\TLCN\Main\datasets\FMFCCV/real_rename"  # Thay bằng đường dẫn thư mục của bạn
output_file = "random_files.txt"    # Tên file output
random_files_to_txt(input_folder, output_file)