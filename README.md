## Mô tả project
Đây là repository chứa source code của Tiểu luận chuyên ngành đề tài "Tìm hiểu và ứng dụng nhận diện deepfake"

## Thành viên thực hiện
| **STT** | **Họ tên**        | **MSSV** | **Ghi chú** |   |
|---------|-------------------|----------|-------------|---|
| 1       | Nguyễn Quốc Lân   | 21110837 | Nhóm trưởng |   |
| 2       | Đinh Đại Hải Đăng | 21110164 |             |   |

## Dataset
Sử dụng CelebDFv2 và FacesForensics++. Để tải dữ liệu, sử dụng file `download-FaceForensics.py` để tải.

## Chương trình Demo
Chạy file `mainUI.py` trong thư mục `training`. Đảm bảo các file yaml trong thư mục `training\config\detector` được thiết lập đúng.  
File `*.pth` tải về ở [đây](https://drive.google.com/drive/folders/1RvU86NZLiGQy78_C4LPcFhL2eAW8cDtE?usp=sharing) và nằm trong thư mục `training\weights`.

## Thư viện cần thiết để sử dụng
Cài đặt sử dụng conda. Chạy câu lệnh
```
conda create -n <environment-name> --file req.txt
```