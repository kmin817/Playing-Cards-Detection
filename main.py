import argparse
import cv2
import sys
import os
import logging

os.environ["YOLO_VERBOSE"] = "False"

logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
from ultralytics import YOLO

DEBUG_MODE = False

def parse_args():
    p = argparse.ArgumentParser("CV assignment runner")

    # Sample argument usage
    p.add_argument("--input", required=True, type=str, help="path to input image")
    p.add_argument("--output", required=True, type=str, help="path to output image")

    return p.parse_args()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input image") 
    parser.add_argument("--task", type=str, required=True, choices=['presence', 'bbox'], help="Task type")
    args = parser.parse_args()

    image_path = args.input
    task = args.task

    # 예외 처리
    if not os.path.exists(image_path):
        return

    # 모델 로드
    try:
        model = YOLO('best.pt')
    except Exception as e:
        sys.exit(1)

    # 추론
    results = model.predict(source=image_path, save=DEBUG_MODE, conf=0.7, verbose=False)

    if DEBUG_MODE and results:
        print(f"\n[DEBUG] 박스가 그려진 이미지가 저장되었습니다: {results[0].save_dir}")
    
    # 결과가 없거나 비어있는 경우 처리
    if not results:
        print("false" if task == 'presence' else "none")
        return

    result = results[0]
    
    # 결과 출력
    if task == 'presence':
        if len(result.boxes) > 0:
            print("true")
        else:
            print("false")

    elif task == 'bbox':
        if len(result.boxes) == 0:
            print("none")
        else:
            box = result.boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            x = int(round(x1))
            y = int(round(y1))
            
            w = int(round(x2 - x1))
            h = int(round(y2 - y1))
            
            print(f"{x},{y},{w},{h}")

if __name__ == "__main__":
    sys.exit(main())
