import os

def rename_files_in_directory(directory_path):
    # Duyệt qua tất cả file trong thư mục
    for filename in os.listdir(directory_path):
        # Tách phần tên và đuôi file
        base_name, ext = os.path.splitext(filename)
        
        if '_' in base_name:  # Kiểm tra nếu là file có dạng số_số_số
            parts = base_name.split('_')
            
            # Bỏ 3 chữ số đầu (phần đầu tiên)
            remaining_parts = parts[1:]
            
            # Ánh xạ các phần số còn lại bằng phép toán
            new_parts = []
            for part in remaining_parts:
                if part.isdigit() and len(part) == 3:  # Chỉ xử lý số có 3 chữ số
                    new_num = int(part) + 999  # 001 + 999 = 1000, 002 + 999 = 1001,...
                    new_parts.append(f"{new_num:04d}")  # Định dạng 4 chữ số
                else:
                    new_parts.append(part)  # Giữ nguyên nếu không phải số 3 chữ số
            
            # Tạo tên mới
            new_base_name = '_'.join(new_parts)
            new_filename = new_base_name + ext
            
            # Đổi tên file
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_filename)
            
            os.rename(old_path, new_path)
            print(f"Đã đổi tên: {filename} -> {new_filename}")

# Sử dụng
directory_path = "E:\TLCN\Main\datasets\FMFCCV/fake_rename/ppk"
rename_files_in_directory(directory_path)