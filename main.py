import argparse

import cv2
import sys
import os
import numpy as np
import math

import labeling2
from src import process
import pastprocess as legacy
from src.utils.Loader import Loader

USE_WINDOWS = False 
DETECT_SCALE = 0.5      
RUN_CLASSIFICATION = True

def parse_args():
    p = argparse.ArgumentParser("CV assignment runner")

    # Sample argument usage
    p.add_argument("--input", required=True, type=str, help="path to input image")
    # p.add_argument("--output", required=True, type=str, help="path to output image")

    return p.parse_args()

def to_3ch(img):
    """단일 채널 이미지를 BGR 3채널로 변환"""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def hstack_images(images, pad=8, pad_color=(30, 30, 30)):
    if not images: return None
    imgs = [to_3ch(im) for im in images]
    min_h = min(im.shape[0] for im in imgs)
    
    # 높이 맞추기 리사이즈
    resized_imgs = []
    for im in imgs:
        ih, iw = im.shape[:2]
        if ih != min_h:
            scale = min_h / ih
            im = cv2.resize(im, (int(iw * scale), min_h), interpolation=cv2.INTER_AREA)
        resized_imgs.append(im)

    row = []
    for i, im in enumerate(resized_imgs):
        row.append(im)
        if i < len(resized_imgs) - 1:
            row.append(np.full((min_h, pad, 3), pad_color, dtype=np.uint8))
    return cv2.hconcat(row)

def vstack_images(rows, pad=8, pad_color=(30, 30, 30)):
    if not rows: return None
    widths = [r.shape[1] for r in rows]
    max_w = max(widths)
    out_rows = []
    for r in rows:
        h, w = r.shape[:2]
        if w < max_w:
            right_pad = np.full((h, max_w - w, 3), (30, 30, 30), dtype=np.uint8)
            r = cv2.hconcat([r, right_pad])
        out_rows.append(r)

    col = []
    for i, r in enumerate(out_rows):
        col.append(r)
        if i < len(out_rows) - 1:
            col.append(np.full((pad, max_w, 3), pad_color, dtype=np.uint8))
    return cv2.vconcat(col)

def analyze_contours(thresh_img, original_bgr, method_name="Otsu"):
    h, w = thresh_img.shape[:2]
    img_area = h * w
    area_thresh = img_area * 0.01 

    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big_cnts = [cnt for cnt in contours if cv2.contourArea(cnt) > area_thresh]
    
    # 디버깅용 이미지 (USE_WINDOWS=False면 표시되지 않음)
    dbg = original_bgr.copy() if USE_WINDOWS else None

    valid_card_count = 0
    overlap_suspected = False
    glare_detected = False 
    border_margin = 10

    for idx, cnt in enumerate(big_cnts):
        x, y, cw, ch = cv2.boundingRect(cnt)
        
        # 1. 조명(테두리 접촉) 검사
        is_touching_border = (x <= border_margin) or (y <= border_margin) or \
                            (x + cw >= w - border_margin) or (y + ch >= h - border_margin)
        
        if is_touching_border:
            glare_detected = True 
            continue

        # 2. 형상 분석
        peri = cv2.arcLength(cnt, True)
        approx_loose = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        v_loose = len(approx_loose)
        
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        is_convex = cv2.isContourConvex(approx_loose)

        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        
        if rw == 0 or rh == 0: ratio = 0
        else:
            short_side = min(rw, rh)
            long_side = max(rw, rh)
            ratio = short_side / long_side
        
        is_normal_ratio = (0.5 <= ratio <= 0.85)
        is_single_candidate = (v_loose == 4) and is_convex and (solidity > 0.90) and is_normal_ratio
        
        if is_single_candidate:
            is_card_shape = True
        else:
            is_card_shape = False
        
        valid_card_count += 1
        if not is_card_shape:
            overlap_suspected = True
            
    return valid_card_count, overlap_suspected, dbg, glare_detected

def decide_mode(original_bgr):
    """
    [하이브리드 모드 결정]
    """
    thresh_otsu = legacy.get_thresh_otsu(original_bgr)
    
    cnt_otsu, overlap_otsu, dbg_otsu, glare_otsu = analyze_contours(thresh_otsu, original_bgr, "Otsu")
    
    # 성공 조건 강화: 카드를 찾았더라도, 조명이 없어야 한다.
    if cnt_otsu > 0 and not glare_otsu:
        if overlap_otsu:
            return "overlap", thresh_otsu
        else:
            return "non_overlap", thresh_otsu

    # 3. Otsu 실패 또는 '조명 감지' 시 Canny 시도
    thresh_canny = legacy.get_thresh_canny(original_bgr)
    
    cnt_canny, overlap_canny, dbg_canny, _ = analyze_contours(thresh_canny, original_bgr, "Canny (Fallback)")
    
    if cnt_canny == 0:
        return "overlap", thresh_canny 
        
    if overlap_canny:
        return "overlap", thresh_canny
    else:
        return "non_overlap", thresh_canny

def detect_cards_non_overlap(original_bgr, original_rgb):
    img_draw = original_bgr.copy() if USE_WINDOWS else None
    h, w = original_bgr.shape[:2]

    # 1. 우선 Otsu(Blob) 방식으로 시도
    thresh_otsu = legacy.get_thresh_otsu(original_bgr)
    
    contours, _ = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    otsu_failed = False
    margin = 10
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h * w * 0.01): continue 
        
        x, y, cw, ch = cv2.boundingRect(cnt)
        is_touching_border = (x <= margin) or (y <= margin) or \
                            (x + cw >= w - margin) or (y + ch >= h - margin)
        
        if is_touching_border:
            otsu_failed = True
            break

    if not otsu_failed:
        corners_list = legacy.find_corners_set(thresh_otsu, img_draw, draw=False)
        if len(corners_list) == 0:
            otsu_failed = True
    
    # 2. Otsu 실패 시 -> Canny로 재시도
    if otsu_failed:
        img_draw = original_bgr.copy() if USE_WINDOWS else None
        thresh_canny = legacy.get_thresh_canny(original_bgr) 
        corners_list = legacy.find_corners_set(thresh_canny, img_draw, draw=False) # draw=False로 변경하여 imshow 방지
    
    return corners_list

def main():
    # Example code
    args = parse_args()
    card_path = args.input # 사용자가 입력한 경로

    original_bgr = cv2.imread(card_path)
    if original_bgr is None:
        # 이미지를 못 찾았을 경우 에러 처리 (빈 출력 방지)
        return 0

    RANK_ORDER = {
        "A": 1, "2": 2, "3": 3, "4": 4, "5": 5,
        "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
        "J": 11, "Q": 12, "K": 13, "Unknown": 99,
    }
    SUIT_ORDER = {
        "Clubs": 0, "Diamonds": 1, "Hearts": 2, "Spades": 3, "Unknown": 99,
    }
    detected_cards = []

    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    # 2. 모드 결정 및 실행
    mode, _ = decide_mode(original_bgr)

    if mode == "overlap":
        # labeling2.py 내부에서도 imshow가 모두 제거되어 있어야 합니다.
        labeling2.run_overlap_pipeline(card_path)
        return 0
    
    # 3. 비겹침(Non-overlap) 처리 파이프라인
    four_corners_set = detect_cards_non_overlap(original_bgr, original_rgb)
    flatten_card_set = process.find_flatten_cards(original_rgb, four_corners_set)

    if RUN_CLASSIFICATION:
        cropped_images = process.get_corner_snip(flatten_card_set)

        ranksuit_list = []
        for i, (img_bin, img_original_gray) in enumerate(cropped_images):
            drawable = img_bin.copy()
            d2 = img_original_gray.copy()

            contours, _ = cv2.findContours(drawable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts_sort = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            cnts_sort = sorted(cnts_sort, key=lambda x: cv2.boundingRect(x)[1])

            ranksuit = []
            for k, cnt in enumerate(cnts_sort):
                x, y, w, h = cv2.boundingRect(cnt)
                x2, y2 = x + w, y + h
                crop = d2[y:y2, x:x2]
                if k == 0:   # rank
                    crop = cv2.resize(crop, (70, 125), 0, 0)
                else:        # suit
                    crop = cv2.resize(crop, (70, 100), 0, 0)
                _, crop = cv2.threshold(crop, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                crop = cv2.bitwise_not(crop)
                ranksuit.append(crop)

            ranksuit_list.append(ranksuit)

        train_ranks = Loader.load_ranks("assets/imgs/ranks")
        train_suits = Loader.load_suits("assets/imgs/suits")

        for i, it in enumerate(ranksuit_list):
            if len(it) < 2:
                continue
            rank_img, suit_img = it[0], it[1]

            rank_name, suit_name = process.template_matching(
                rank_img, suit_img, train_ranks, train_suits, show_plt=False
            )

            if rank_name == "Q3": rank_name = "Q"
            if suit_name == "Spades2": suit_name = "Spades"

            detected_cards.append((rank_name, suit_name))

        # 정렬
        detected_cards.sort(key=lambda x: (RANK_ORDER.get(x[0], 99), SUIT_ORDER.get(x[1], 99)))

        short_codes = []
        for rank, suit in detected_cards:
            if rank == "Unknown" or suit == "Unknown":
                # short_codes.append("??")
                continue
            code = f"{suit[0]}{rank}"
            short_codes.append(code)

        # 최종 출력 (표준 출력)
        print(" ".join(short_codes))
    
    # End of example code
    return 0

if __name__ == "__main__":
    sys.exit(main())
