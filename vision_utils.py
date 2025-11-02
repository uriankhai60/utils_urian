from PIL import Image
import numpy as np
import cv2
import dlib

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".avi", ".mp4"}


def get_boundingbox(face, width, height):
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * 1.3)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb


def detect_and_crop_face_optimized(image: Image.Image, target_size=(224, 224), resize_for_detection=640):
    if image.mode != 'RGB': image = image.convert('RGB')
    original_np = np.array(image)
    original_h, original_w, _ = original_np.shape
    if original_w > resize_for_detection:
        scale = resize_for_detection / float(original_w)
        resized_h = int(original_h * scale)
        resized_np = cv2.resize(original_np, (resize_for_detection, resized_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        resized_np = original_np
    
    # dlib detector는 전역 변수로 사용하지 않고, 각 프로세스에서 새로 생성하는 것이 안전합니다.
    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(resized_np, 1)

    if not faces: return None
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    scaled_face_rect = dlib.rectangle(
        left=int(face.left() / scale), top=int(face.top() / scale),
        right=int(face.right() / scale), bottom=int(face.bottom() / scale)
    )
    x, y, size = get_boundingbox(scaled_face_rect, original_w, original_h)
    cropped_np = original_np[y:y + size, x:x + size]
    face_img = Image.fromarray(cropped_np).resize(target_size, Image.BICUBIC)
    return face_img

def process_single_file(file_path):
    """파일 경로를 입력받아 전처리된 얼굴 이미지 리스트와 파일 이름을 반환"""
    face_images = []
    ext = file_path.suffix.lower()
    num_frames_to_extract = 30

    try:
        if ext in IMAGE_EXTS:
            image = Image.open(file_path)
            face_img = detect_and_crop_face_optimized(image)
            if face_img:
                face_images.append(face_img)
        elif ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(str(file_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                frame_indices = np.linspace(0, total_frames - 1, num_frames_to_extract, dtype=int)
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret: continue
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    face_img = detect_and_crop_face_optimized(image)
                    if face_img:
                        face_images.append(face_img)
            cap.release()
    except Exception as e:
        # 오류가 발생한 파일 경로와 오류 메시지를 반환하여 디버깅에 활용할 수 있습니다.
        return file_path.name, [], str(e)

    return file_path.name, face_images, None