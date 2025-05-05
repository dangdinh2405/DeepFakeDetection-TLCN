from moviepy.editor import VideoFileClip
import os

def process_video(input_path, output_path, duration=30):
    """
    Cắt video còn 20 giây và nén theo chuẩn C23
    
    :param input_path: Đường dẫn video gốc
    :param output_path: Đường dẫn video đầu ra
    :param duration: Thời lượng cần giữ lại (giây)
    """
    try:
        # Load video
        clip = VideoFileClip(input_path)
        
        # Cắt video còn 20 giây đầu tiên
        if clip.duration > duration:
            clip = clip.subclip(0, duration)
        
        # Nén theo chuẩn C23 (H.264 với bitrate ~2500kbps)
        clip.write_videofile(
            output_path,
            codec='libx264',
            bitrate='2500k',
            audio_codec='aac',
            preset='medium',
            threads=4,
            ffmpeg_params=['-profile:v', 'high', '-pix_fmt', 'yuv420p']
        )
        
        print(f"Đã xử lý thành công: {os.path.basename(input_path)}")
        
    except Exception as e:
        print(f"Lỗi khi xử lý {input_path}: {str(e)}")
    finally:
        if 'clip' in locals():
            clip.close()

# Sử dụng
input_folder = "E:\TLCN\Main\datasets\FMFCCV/fake_rename"
output_folder = "E:\TLCN\Main\datasets\FMFCCV/fake_30s"

# Tạo thư mục đầu ra nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Xử lý tất cả video trong thư mục
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_video(input_path, output_path, 15)