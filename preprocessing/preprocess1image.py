import cv2
import dlib
from pathlib import Path
import numpy as np
from imutils import face_utils
from skimage import transform as trans

# Tải mô hình nhận diện khuôn mặt
face_detector = dlib.get_frontal_face_detector()
predictor_path = './preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat'
face_predictor = dlib.shape_predictor(predictor_path)

def get_keypts(image, face, predictor, face_detector):
    # Nhận diện các điểm đặc trưng trên khuôn mặt
    shape = predictor(image, face)
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)
    return pts

def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize
        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # Cắt khuôn mặt
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))

        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img, None

    # Chuyển đổi ảnh sang RGB và nhận diện khuôn mặt
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb, 1)

    if len(faces):
        # Chọn khuôn mặt lớn nhất
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        landmarks = get_keypts(rgb, face, predictor, face_detector)
        cropped_face, mask_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        return cropped_face, landmarks, mask_face
    else:
        return None, None, None

# Đọc ảnh và xử lý
image_path = 'E:/TLCN/Main/training/imagetest/2.png'
image = cv2.imread(image_path)

cropped_face, landmarks, mask = extract_aligned_face_dlib(face_detector, face_predictor, image, res=256)

output_path = 'E:/TLCN/Main/training/imagetest/2_pre.png'
cv2.imwrite(output_path, cropped_face)

print(f"Image saved at {output_path}")
