# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
from pathlib import Path
import numpy as np
import os
from typing import List

face_det_checkpoint_path = "model/YOLOv8-Face-Detection/model.pt"
face_det_model = YOLO(face_det_checkpoint_path)

def get_top1_bbox(bboxes:List[List])->List:
    """
    들어온 2차 리스트형태의 bboxes에서 top1 좌표만 리턴하는 함수
    """
    coord = None
    if bboxes is None or len(bboxes) == 0:
        return coord
    # 각 얼굴의 면적 계산 (너비 * 높이)
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    # 가장 큰 얼굴의 인덱스 찾기
    largest_idx = np.argmax(areas)
    # 가장 큰 얼굴의 좌표 반환
    coord = bboxes[largest_idx]
    return coord

def crop_image_by_bbox(canvas: Image.Image, bbox: List, pad: float) -> Image.Image:
    """
    캔버스에서 bbox의 영역만 크롭 후 리턴하는 함수
    """
    # 캔버스 크기 가져오기 (경계 체크용)
    canvas_width, canvas_height = canvas.size
    
    # bbox 좌표
    xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    
    # bbox의 width, height 계산
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin
    
    # bbox 크기의 비율만큼 패딩 픽셀 계산
    pad_x = int(bbox_width * pad)
    pad_y = int(bbox_height * pad)
    
    # 패딩 적용 (이미지 경계를 넘지 않도록 clamp)
    xmin_padded = max(0, xmin - pad_x)
    ymin_padded = max(0, ymin - pad_y)
    xmax_padded = min(canvas_width, xmax + pad_x)
    ymax_padded = min(canvas_height, ymax + pad_y)
    
    # 크롭
    cropped_canvas = canvas.crop((xmin_padded, ymin_padded, xmax_padded, ymax_padded))
    return cropped_canvas

if __name__ == "__main__":
    output_dir = Path("outs_face_det")
    output_dir.mkdir(parents=True, exist_ok=True)

    filepaths = sorted(list(Path("data").rglob("*.png")))
    images = [Image.open(filepath) for filepath in filepaths]
    
    for idx, (input_image, filepath) in enumerate(zip(images,filepaths)):
        output = face_det_model(input_image, verbose=False, conf=0.5)  # 이미지1개 입력 결과 1개
        results = Detections.from_ultralytics(output[0])  # 첫번째 결과
        bboxes = results.xyxy
        bbox = get_top1_bbox(bboxes)
        
        if bbox is None:
            input_image.save(output_dir/os.path.basename(filepath))
            continue

        crop_image = crop_image_by_bbox(input_image, bbox, 0.2)
        crop_image.save(output_dir/os.path.basename(filepath))

