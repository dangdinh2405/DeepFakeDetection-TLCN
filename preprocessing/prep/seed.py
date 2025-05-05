import os
import json
import random
from itertools import combinations

random.seed(42)

# Đường dẫn chứa video
video_dir = "E:\\TLCN\\Main\\datasets\\FaceForensics++\\original_sequences\\youtube\\c23\\videos"
all_video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
video_ids = [f.split('.')[0] for f in all_video_files]

# Kiểm tra trùng lặp
assert len(video_ids) == len(set(video_ids)), "Có video ID trùng lặp!"

# Xáo trộn thứ tự
random.shuffle(video_ids)

# Tỷ lệ chia
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Chia video thành các tập
total_videos = len(video_ids)
train_count = int(total_videos * train_ratio)
val_count = int(total_videos * val_ratio)
test_count = total_videos - train_count - val_count

train_videos = video_ids[:train_count]
val_videos = video_ids[train_count:train_count+val_count]
test_videos = video_ids[train_count+val_count:]

# Hàm tạo các cặp không trùng lặp
def generate_unique_pairs(video_list, max_pairs=None):
    """Tạo các cặp sao cho mỗi video chỉ xuất hiện một lần"""
    pairs = []
    used_videos = set()
    
    # Tạo tất cả cặp có thể
    all_possible = list(combinations(video_list, 2))
    random.shuffle(all_possible)
    
    for pair in all_possible:
        # Kiểm tra nếu video trong cặp chưa được sử dụng
        if pair[0] not in used_videos and pair[1] not in used_videos:
            pairs.append(pair)
            used_videos.update(pair)
            
            # Dừng nếu đạt đủ số lượng cặp
            if max_pairs and len(pairs) >= max_pairs:
                break
                
    return pairs

# Giới hạn số cặp để đảm bảo không trùng lặp
max_possible_train_pairs = len(train_videos) // 2
max_possible_val_pairs = len(val_videos) // 2
max_possible_test_pairs = len(test_videos) // 2

train_pairs = generate_unique_pairs(train_videos, max_possible_train_pairs)
val_pairs = generate_unique_pairs(val_videos, max_possible_val_pairs)
test_pairs = generate_unique_pairs(test_videos, max_possible_test_pairs)

# Kiểm tra
def check_uniqueness(pairs):
    video_counts = {}
    for pair in pairs:
        for vid in pair:
            video_counts[vid] = video_counts.get(vid, 0) + 1
    return all(count == 1 for count in video_counts.values())

assert check_uniqueness(train_pairs)
assert check_uniqueness(val_pairs)
assert check_uniqueness(test_pairs)

# Lưu kết quả
output_dir = "unique_pairs_splits"
os.makedirs(output_dir, exist_ok=True)

def save_data(data, filename):
    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump([list(pair) for pair in data], f, indent=2)

save_data(train_pairs, "train_pairs.json")
save_data(val_pairs, "val_pairs.json")
save_data(test_pairs, "test_pairs.json")

# Lưu video assignments
assignments = {}
for vid in train_videos:
    assignments[vid] = "train" if vid in [v for pair in train_pairs for v in pair] else "unused_train"
for vid in val_videos:
    assignments[vid] = "val" if vid in [v for pair in val_pairs for v in pair] else "unused_val"
for vid in test_videos:
    assignments[vid] = "test" if vid in [v for pair in test_pairs for v in pair] else "unused_test"

with open(os.path.join(output_dir, "video_assignments.json"), 'w') as f:
    json.dump(assignments, f, indent=2)

# Thống kê
stats = {
    "total_videos": total_videos,
    "train_videos": len(train_videos),
    "val_videos": len(val_videos),
    "test_videos": len(test_videos),
    "used_train_videos": len(set([v for pair in train_pairs for v in pair])),
    "used_val_videos": len(set([v for pair in val_pairs for v in pair])),
    "used_test_videos": len(set([v for pair in test_pairs for v in pair])),
    "train_pairs": len(train_pairs),
    "val_pairs": len(val_pairs),
    "test_pairs": len(test_pairs),
    "unused_train_videos": len(train_videos) - len(set([v for pair in train_pairs for v in pair])),
    "unused_val_videos": len(val_videos) - len(set([v for pair in val_pairs for v in pair])),
    "unused_test_videos": len(test_videos) - len(set([v for pair in test_pairs for v in pair]))
}

with open(os.path.join(output_dir, "stats.json"), 'w') as f:
    json.dump(stats, f, indent=2)

print("Kết quả:")
print(f"- Train: {len(train_pairs)} cặp (sử dụng {stats['used_train_videos']}/{len(train_videos)} video)")
print(f"- Val: {len(val_pairs)} cặp (sử dụng {stats['used_val_videos']}/{len(val_videos)} video)")
print(f"- Test: {len(test_pairs)} cặp (sử dụng {stats['used_test_videos']}/{len(test_videos)} video)")
print(f"Đã lưu vào thư mục: {output_dir}")