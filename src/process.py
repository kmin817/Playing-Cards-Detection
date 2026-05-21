import cv2
import numpy as np
import math

from src.ColorHelper import ColorHelper
from src.utils.DistanceHelper import DistanceHelper

CARD_RATIO = 1.45 

# 1) 에지/바이너리 생성
def get_thresh(img):
    """
    카드 경계를 잘 살리기 위한 전처리 + Canny.
    img: BGR 이미지
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 2)
    # canny = cv2.Canny(blur, 42, 120)
    canny = cv2.Canny(blur, 100, 200)
    kernel = np.ones((2, 2), np.uint8)
    dial = cv2.dilate(canny, kernel=kernel, iterations=1)
    return dial


# 2) 카드 사각형 검출
def _rect_area(rect):
    (_, _), (w, h), _ = rect
    return w * h


def _nms_rotated_rects(rects, dist_thresh=25.0, angle_thresh=15.0):
    if not rects:
        return []

    # strength(면적) 내림차순 정렬
    rects = sorted(rects, key=lambda x: x[1], reverse=True)

    picked = []
    for rect, strength in rects:
        (cx, cy), (wh, hh), angle = rect
        keep = True
        for prect, pstrength in picked:
            (pcx, pcy), (pwh, phh), pangle = prect

            dist = np.hypot(pcx - cx, pcy - cy)
            adiff = abs(pangle - angle)
            # angle wrap-around 보정 (예: -89도 vs 91도)
            adiff = min(adiff, 180.0 - adiff)

            if dist < dist_thresh and adiff < angle_thresh:
                keep = False
                break

        if keep:
            picked.append((rect, strength))

    return [rect for rect, _ in picked]

def _order_box_points(box):
    """
    box: cv2.boxPoints() 결과 (4x2)
    반환: [tl, bl, br, tr] 순서 (float32)
    """
    pts = np.array(box, dtype=np.float32)

    # (x + y)가 가장 작은 -> TL, 가장 큰 -> BR
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    # (x - y)가 가장 작은 -> BL, 가장 큰 -> TR
    diff = pts[:, 0] - pts[:, 1]
    bl = pts[np.argmin(diff)]
    tr = pts[np.argmax(diff)]

    ordered = np.array([tl, bl, br, tr], dtype=np.float32)

    # 카드가 가로로 누운 경우(width > height)면 회전 보정
    h = np.linalg.norm(ordered[0] - ordered[1])
    w = np.linalg.norm(ordered[0] - ordered[3])
    if w > h:
        # TL, BL, BR, TR -> (BL, BR, TR, TL)로 한 칸씩 밀어 회전
        ordered = np.array([ordered[1], ordered[2], ordered[3], ordered[0]], dtype=np.float32)

    return ordered


def find_corners_set(edge_img, original, draw=True, debug=False,
                    min_area=2000, aspect_low=0.9, aspect_high=2.2):
    """
    edge_img : get_thresh 결과 (단일 채널)
    original : 디버그용으로 사각형을 그릴 BGR 이미지 (검출 해상도 기준)
    반환값  : four_corners_set
    각 원소는 [[[x,y]], [[x,y]], [[x,y]], [[x,y]]] 형식
    순서는 [top-left, bottom-left, bottom-right, top-right]
    """
    # 1) 외곽선 찾기
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = edge_img.shape[:2]
    dbg_all = original.copy()
    dbg_final = original.copy()

    candidates = []
    # print(f"[DEBUG] Total contours: {len(contours)}")

    # 2) 모든 contour에 대해 minAreaRect로 사각형 후보 생성
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area <min_area:
            # 너무 작은 노이즈만 제외 (디버그는 해봄)
            if area > 500:
                x, y, ww, hh = cv2.boundingRect(cnt)
                # print(f"[DEBUG] Small contour {i}: area={area:.1f}, bbox=({x},{y},{ww},{hh})")
            continue

        rect = cv2.minAreaRect(cnt)  # ((cx, cy), (w, h), angle)
        (cx, cy), (rw, rh), angle = rect

        if rw < 1 or rh < 1:
            continue

        # width >= height가 되도록 강제
        if rw < rh:
            rw, rh = rh, rw
            angle += 90.0
            rect = ((cx, cy), (rw, rh), angle)

        aspect = rw / rh

        # 카드 비율 근처만 허용 (1.45 근처 널널하게)
        if not (aspect_low <= aspect <= aspect_high):
            x, y, ww, hh = cv2.boundingRect(cnt)
            # print(f"[REJECT] Contour {i}: area={area:.1f}, aspect={aspect:.2f}, bbox=({x},{y},{ww},{hh})")
            continue

        # strength는 일단 area로 설정
        strength = area
        candidates.append((rect, strength))

        # 후보 사각형은 얇은 노란색으로 전체 디버그 그림에 표시
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(dbg_all, [box], 0, (0, 255, 255), 2)

    # print(f"[INFO] Candidate card rectangles (before NMS): {len(candidates)}")

    # 3) NMS로 겹치는 후보 정리
    picked_rects = _nms_rotated_rects(candidates, dist_thresh=25.0, angle_thresh=15.0)
    # print(f"[INFO] Final card rectangles (after NMS): {len(picked_rects)}")

    four_corners_set = []

    # 4) 최종 rect들을 TL/BL/BR/TR 순서로 정리하고, 굵은 초록색으로 그리기
    for idx, rect in enumerate(picked_rects):
        box = cv2.boxPoints(rect)
        ordered = _order_box_points(box)  # (4,2) float32

        # original에 그리기
        ibox = np.int32(ordered)
        # cv2.drawContours(dbg_final, [ibox], 0, (0, 255, 0), 3)
        # cv2.putText(
        #     dbg_final,
        #     f"#{idx}",
        #     (int(ordered[0][0]), int(ordered[0][1]) - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.6,
        #     (0, 255, 0),
        #     2,
        #     cv2.LINE_AA,
        # )

        # 예전 인터페이스와 동일하게 [[[x,y]], ...] 형식으로 변환
        finalOrder = [[p.tolist()] for p in ordered]
        four_corners_set.append(finalOrder)

        # print(f"[CARD {idx}] corners (TL,BL,BR,TR): {ordered.tolist()}")

    # 5) 디버그용 이미지 두 장 리턴 대신, original을 수정한 형태로 사용하도록 설계
    if draw and debug:
        # dbg_all: 모든 후보(노란색)
        # dbg_final: 최종 선택 카드(초록색)
        # cv2.namedWindow("DEBUG_all_candidates", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("DEBUG_final_cards", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("DEBUG_all_candidates", w // 2, h // 2)
        # cv2.resizeWindow("DEBUG_final_cards", w // 2, h // 2)
        # cv2.imshow("DEBUG_all_candidates", dbg_all)
        # cv2.imshow("DEBUG_final_cards", dbg_final)
        # cv2.waitKey(0)
        # cv2.destroyWindow("DEBUG_all_candidates")
        # cv2.destroyWindow("DEBUG_final_cards")
        pass
    elif draw:
        # draw=True, debug=False면 final만 original에 그려진 상태를 main에서 imshow
        original[:] = dbg_final

    return four_corners_set


# ---------------------------
# 2-1) 스케일 보정용 헬퍼
# ---------------------------
def scale_corners(corners_set, scale):
    """
    다운스케일된 이미지에서 얻은 코너 좌표를
    원본 스케일로 되돌릴 때 사용.

    corners_set: find_corners_set 반환값
    scale      : 배율 (예: DETECT_SCALE=0.5로 줄였다면 scale=1/0.5=2.0)
    """
    scaled = []
    for card in corners_set:
        new_card = []
        for pt in card:
            x, y = pt[0]
            sx = int(round(x * scale))
            sy = int(round(y * scale))
            new_card.append([[sx, sy]])
        scaled.append(new_card)
    return scaled


# 3) 카드 평탄화
def find_flatten_cards(img, set_of_corners, debug=False):
    width, height = 200, 300
    img_outputs = []

    for i, corners in enumerate(set_of_corners):
        top_left = corners[0][0]
        bottom_left = corners[1][0]
        bottom_right = corners[2][0]
        top_right = corners[3][0]

        _vertical_left = DistanceHelper.euclidean(top_left[0], top_left[1], bottom_left[0], bottom_left[1])
        _horizontal_top = DistanceHelper.euclidean(top_left[0], top_left[1], top_right[0], top_right[1])

        pts1 = np.float32([top_left, bottom_left, bottom_right, top_right])
        pts2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_output = cv2.warpPerspective(img, matrix, (width, height))
        img_outputs.append(img_output)

    return img_outputs


# 4) 코너 스니핑 
def get_corner_snip(flattened_images: list):
    corner_images = []
    for img in flattened_images:
        crop = img[5:110, 1:38]

        crop = cv2.resize(crop, None, fx=4, fy=4)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bin_img = ColorHelper.gray2bin(gray)
        bilateral = cv2.bilateralFilter(bin_img, 11, 174, 17)
        canny = cv2.Canny(bilateral, 40, 24)
        kernel = np.ones((1, 1))
        result = cv2.dilate(canny, kernel=kernel, iterations=2)

        corner_images.append([result, bin_img])

    return corner_images


def split_rank_suit(img, original, debug=False) -> list:
    """
    :param debug: display opencv or not
    :param img:
    :param original: original image
    :return: list of image, index 0: rank, index 1: suit
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts_sort = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    cnts_sort = sorted(cnts_sort, key=lambda x: cv2.boundingRect(x)[1])

    # cv2.drawContours(img, cnts_sort, -1, (0, 255, 0), 1)

    ranksuit = []
    _rank = None

    for i, cnt in enumerate(cnts_sort):
        x, y, w, h = cv2.boundingRect(cnt)
        x2, y2 = x + w, y + h
        crop = original[y:y2, x:x2]

        if i == 0:
            crop = cv2.resize(crop, (70, 125), 0, 0)
            _rank = crop
        else: 
            crop = cv2.resize(crop, (70, 100), 0, 0)
            if debug and _rank is not None:
                pass
                # r = cv2.resize(_rank, (70, 100), 0, 0)
                # s = cv2.resize(crop, (70, 100), 0, 0)
                # hcat = np.concatenate((r, s), axis=1)
                # hcat = cv2.resize(hcat, (250, 200), 0, 0)
                # cv2.imshow("crop2", hcat)
                # cv2.waitKey(1)

        crop = ColorHelper.gray2bin(crop)
        crop = ColorHelper.reverse(crop)
        ranksuit.append(crop)

    return ranksuit


def template_matching(rank, suit, train_ranks, train_suits, show_plt=False) -> tuple[str, str]:
    best_rank_match_diff = 10000
    best_suit_match_diff = 10000
    best_rank_match_name = "Unknown"
    best_suit_match_name = "Unknown"

    best_rank_diff_img = None
    best_suit_diff_img = None

    # Rank
    for train_rank in train_ranks:
        diff_img = cv2.absdiff(rank, train_rank.img)
        rank_diff = int(np.sum(diff_img) / 255)

        if rank_diff < best_rank_match_diff:
            best_rank_match_diff = rank_diff
            best_rank_match_name = train_rank.name
            best_rank_diff_img = diff_img.copy()

            if show_plt:
                pass
                # vis = cv2.hconcat([
                #     _to3c(rank),
                #     _to3c(train_rank.img),
                #     _to3c(best_rank_diff_img)
                # ])
                # cv2.imshow("rank / template / diff(best-so-far)", vis)
                # cv2.waitKey(1)

    # Suit
    for train_suit in train_suits:
        diff_img = cv2.absdiff(suit, train_suit.img)
        suit_diff = int(np.sum(diff_img) / 255)

        if suit_diff < best_suit_match_diff:
            best_suit_match_diff = suit_diff
            best_suit_match_name = train_suit.name
            best_suit_diff_img = diff_img.copy()

            if show_plt:
                pass
                # vis = cv2.hconcat([
                #     _to3c(suit),
                #     _to3c(train_suit.img),
                #     _to3c(best_suit_diff_img)
                # ])
                # cv2.imshow("suit / template / diff(best-so-far)", vis)
                # cv2.waitKey(1)

    # Thresholding rule 유지
    rank_name = best_rank_match_name if best_rank_match_diff < 2300 else "Unknown"
    suit_name = best_suit_match_name if best_suit_match_diff < 1000 else "Unknown"

    if show_plt:
        # cv2.waitKey(0)
        pass

    return rank_name, suit_name


def _to3c(img):
    """단일 채널을 BGR 3채널로 변환(디버그 합성용)."""
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img


def show_text(predictions: list[str], four_corners_set, img):
    for i, prediction in enumerate(predictions):
        corners = np.array(four_corners_set[i])
        corners_flat = corners.reshape(-1, corners.shape[-1])
        start_x = corners_flat[0][0] + 0
        half_y = corners_flat[0][1] - 40

        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(img, prediction, (start_x, half_y), font, 0.8, (50, 205, 50), 2, cv2.LINE_AA)


def eval_rank_suite(rank_suit_mapping, modelRanks, modelSuits):
    """
    (더미 함수)
    원래는 model_wrapper를 사용해 rank/suit 분류 모델을 예측하는 역할이었지만,
    현재 템플릿 매칭 방식으로 대체되어 비활성화 상태입니다.
    """
    # print("[INFO] eval_rank_suite() is disabled (model_wrapper not defined).")
    # print(f"[DEBUG] Received {len(rank_suit_mapping)} rank/suit pairs but skipping evaluation.")
    return []

def detect_cards_hough(gray_det, edge_img, card_w, card_h,
                    angle_step_deg=2,   
                    vote_blur_xy=5,     
                    vote_thresh=None,
                    debug=False, max_cards=4):
    """
    [최종 솔루션 - macOS 에러 수정본] Iterative Edge Peeling GHT
    """
    h, w = gray_det.shape[:2]
    if card_h < card_w: card_w, card_h = card_h, card_w

    # 1) 파라미터 및 공통 데이터 준비
    R_short = card_w / 2.0 
    R_long = card_h / 2.0
    angle_step = np.deg2rad(angle_step_deg)
    n_phi = int(np.pi / angle_step)
    
    gx = cv2.Sobel(gray_det, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_det, cv2.CV_32F, 0, 1, ksize=3)
    phi_all = np.arctan2(gy, gx)
    
    # 현재 살아있는 에지 이미지 (복사본)
    current_edge_img = edge_img.copy()
    
    four_corners_set = []
    
    # --- [핵심] 반복적 검출 루프 ---
    for i in range(max_cards):
        # 1. 살아있는 에지 좌표 추출
        y_idxs, x_idxs = np.where(current_edge_img > 0)
        N = len(y_idxs)
        
        if N < 100: 
            if debug: 
                # print(f"[GHT-ITER] Stop: Too few edges left ({N})")
                pass
            break
            
        # 2. Vectorized Voting
        phis = phi_all[y_idxs, x_idxs]
        cos_p, sin_p = np.cos(phis), np.sin(phis)
        v_x, v_y = cos_p[:, None], sin_p[:, None]
        w_x, w_y = -sin_p[:, None], cos_p[:, None]
        pts_x, pts_y = x_idxs[:, None], y_idxs[:, None]

        # 밝기 방향 판별
        check_dist = 3
        chk_x_pos = np.clip(pts_x + v_x * check_dist, 0, w-1).astype(int)
        chk_y_pos = np.clip(pts_y + v_y * check_dist, 0, h-1).astype(int)
        chk_x_neg = np.clip(pts_x - v_x * check_dist, 0, w-1).astype(int)
        chk_y_neg = np.clip(pts_y - v_y * check_dist, 0, h-1).astype(int)
        
        val_pos = gray_det[chk_y_pos.ravel(), chk_x_pos.ravel()].reshape(-1, 1)
        val_neg = gray_det[chk_y_neg.ravel(), chk_x_neg.ravel()].reshape(-1, 1)
        signs = np.where(val_pos > val_neg, 1.0, -1.0)

        # 가설 생성 & 투표
        step_sampling = 3
        j_vals = np.arange(-int(R_long), int(R_long) + 1, step_sampling)[None, :]
        k_vals = np.arange(-int(R_short), int(R_short) + 1, step_sampling)[None, :]

        # Long Edge (Move R_short)
        c1_x = pts_x + R_short * v_x * signs + j_vals * w_x
        c1_y = pts_y + R_short * v_y * signs + j_vals * w_y
        
        phi_norm = phis.copy()
        phi_norm[phi_norm < 0] += np.pi
        phi_long = phi_norm + (np.pi / 2.0)
        phi_long[phi_long >= np.pi] -= np.pi
        p1_idx = (np.round(phi_long / angle_step).astype(np.int32) % n_phi)[:, None]
        p1_idx = np.repeat(p1_idx, c1_x.shape[1], axis=1)

        # Short Edge (Move R_long)
        c2_x = pts_x + R_long * v_x * signs + k_vals * w_x
        c2_y = pts_y + R_long * v_y * signs + k_vals * w_y
        
        p2_idx = (np.round(phi_norm / angle_step).astype(np.int32) % n_phi)[:, None]
        p2_idx = np.repeat(p2_idx, c2_x.shape[1], axis=1)

        # Flatten
        all_cx = np.concatenate([c1_x.ravel(), c2_x.ravel()])
        all_cy = np.concatenate([c1_y.ravel(), c2_y.ravel()])
        all_pi = np.concatenate([p1_idx.ravel(), p2_idx.ravel()])

        all_cx = np.rint(all_cx).astype(np.int32)
        all_cy = np.rint(all_cy).astype(np.int32)
        mask = (all_cx >= 0) & (all_cx < w) & (all_cy >= 0) & (all_cy < h)
        
        # Accumulator
        H = np.zeros((h, w, n_phi), dtype=np.uint16)
        np.add.at(H, (all_cy[mask], all_cx[mask], all_pi[mask]), 1)

        # 3. Peak Finding
        H_2d = np.sum(H, axis=2, dtype=np.float32)
        if vote_blur_xy > 0:
            ksize = vote_blur_xy * 2 + 1
            H_2d = cv2.GaussianBlur(H_2d, (ksize, ksize), 0)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(H_2d)
        
        curr_thresh = vote_thresh
        if curr_thresh is None:
            curr_thresh = max(25, max_val * 0.5)
            
        if max_val < curr_thresh:
            if debug: 
                # print(f"[GHT-ITER] Stop: Max vote ({max_val:.1f}) < Thresh ({curr_thresh:.1f})")
                pass
            break
            
        cx, cy = max_loc
        
        y0, y1 = max(0, cy-2), min(h, cy+3)
        x0, x1 = max(0, cx-2), min(w, cx+3)
        local_hist = np.sum(H[y0:y1, x0:x1, :], axis=(0, 1))
        pidx = np.argmax(local_hist)

        if debug:
            # print(f"[GHT-ITER] Card #{i}: center=({cx},{cy}), val={max_val:.1f}")
            pass

        # 4. Box Recovery
        phi_hat = pidx * angle_step
        c = np.array([cx, cy], dtype=np.float32)
        u = np.array([math.cos(phi_hat), math.sin(phi_hat)], dtype=np.float32)
        v_perp = np.array([-math.sin(phi_hat), math.cos(phi_hat)], dtype=np.float32)

        tl = c - u * card_h/2 - v_perp * card_w/2
        bl = c - u * card_h/2 + v_perp * card_w/2
        br = c + u * card_h/2 + v_perp * card_w/2
        tr = c + u * card_h/2 - v_perp * card_w/2

        ordered = np.stack([tl, bl, br, tr], axis=0)
        ordered[:, 0] = np.clip(ordered[:, 0], 0, w - 1)
        ordered[:, 1] = np.clip(ordered[:, 1], 0, h - 1)
        
        finalOrder = [[[int(round(p[0])), int(round(p[1]))]] for p in ordered]
        four_corners_set.append(finalOrder)
        
        # --- 5. Edge Peeling ---
        box_poly = ordered.astype(np.int32)
        cv2.drawContours(current_edge_img, [box_poly], -1, 0, thickness=9) 
        
        # 디버깅용: 지워지고 남은 에지 확인
        if debug:
            # win_name = f"DEBUG_Edge_Left_{i}"
            # cv2.imshow(win_name, cv2.resize(current_edge_img, (0,0), fx=0.5, fy=0.5))
            # cv2.waitKey(1) 
            pass

    if debug:
        pass
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()

    return four_corners_set

def compute_iou(boxA, boxB):
    
    ptsA = np.array(boxA, dtype=np.float32).reshape(4, 2)
    ptsB = np.array(boxB, dtype=np.float32).reshape(4, 2)
    
    # AABB 좌표 구하기
    xA_min, yA_min = np.min(ptsA, axis=0)
    xA_max, yA_max = np.max(ptsA, axis=0)
    xB_min, yB_min = np.min(ptsB, axis=0)
    xB_max, yB_max = np.max(ptsB, axis=0)

    # 교차 영역 (Intersection)
    xI_min = max(xA_min, xB_min)
    yI_min = max(yA_min, yB_min)
    xI_max = min(xA_max, xB_max)
    yI_max = min(yA_max, yB_max)

    interArea = max(0, xI_max - xI_min) * max(0, yI_max - yI_min)

    # 합집합 영역 (Union)
    boxAArea = (xA_max - xA_min) * (yA_max - yA_min)
    boxBArea = (xB_max - xB_min) * (yB_max - yB_min)
    unionArea = boxAArea + boxBArea - interArea

    if unionArea == 0: return 0
    return interArea / unionArea

def filter_ght_candidates(gray_det, edge_img,
                        corners_set,
                        card_w,
                        card_h,
                        max_cards=2,
                        area_tol=0.5,
                        min_edge_density=0.01,
                        debug=False):

    h, w = edge_img.shape[:2]
    target_area = float(card_w * card_h)
    min_area, max_area = target_area * 0.5, target_area * 1.5

    scored = []

    for idx, corners in enumerate(corners_set):
        pts = np.array([p[0] for p in corners], dtype=np.int32)
        area = cv2.contourArea(pts)
        if area < 1: continue

        if not (min_area <= area <= max_area): continue

        mask = np.zeros_like(edge_img, dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)

        edge_in = cv2.bitwise_and(edge_img, edge_img, mask=mask)
        edge_count = int(np.count_nonzero(edge_in))
        density = edge_count / (area + 1e-6)

        gray_in = cv2.bitwise_and(gray_det, gray_det, mask=mask)
        pixel_count = np.count_nonzero(mask)
        if pixel_count == 0: continue
        mean_gray = float(gray_in.sum()) / float(pixel_count)
        global_mean = float(gray_det.mean())

        if mean_gray < global_mean + 10: continue
        if density < min_edge_density: continue

        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())

        scored.append({
            "idx": idx,
            "corners": corners,
            "area": area,
            "density": density,
            "center": (cx, cy),
            "mean_gray": mean_gray,
            "score": -idx
        })

    # 점수 높은 순(idx 작은 순)으로 정렬되어 있다고 가정 (GHT 출력 특성상)
    # 혹시 모르니 idx 기준 오름차순 정렬 (0번이 1등)
    scored.sort(key=lambda x: x["idx"])

    # ---  IoU 기반 NMS ---
    picked = []
    iou_thresh = 0.65 # 30% 이상 겹치면 중복으로 간주하고 제거

    for cand in scored:
        if len(picked) >= max_cards: break

        # 이미 뽑힌 카드들과 IoU 비교
        is_duplicate = False
        
        box_curr = [p[0] for p in cand["corners"]]
        
        for picked_cand in picked:
            box_picked = [p[0] for p in picked_cand["corners"]]
            
            iou = compute_iou(box_curr, box_picked)
            
            if iou > iou_thresh:
                if debug: 
                    # print(f"[GHT-FILTER] Reject #{cand['idx']} (IoU={iou:.2f} with #{picked_cand['idx']})")
                    pass
                is_duplicate = True
                break
        
        if not is_duplicate:
            picked.append(cand)

    if debug:
        # print(f"[GHT-FILTER] picked {len(picked)} cards")
        pass

    return [c["corners"] for c in picked]


# 카드 사이즈 검출
def estimate_card_size(edge_img, 
                        debug_image=None, 
                        min_length=50, 
                        aspect_ratio=1.45, 
                        debug=False):
    """
    [최종 개선 버전]
    1. 노이즈 제거 (Cleaning)
    2. 모폴로지 닫기 (Closing) -> 문양으로 끊어진 선 연결
    3. 동적 파라미터 (Dynamic Hough) -> 해상도 맞춤형 Gap 설정
    """
    h, w = edge_img.shape[:2]
    diag = np.hypot(w, h)
    
    # --- 노이즈 청소  ---
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(edge_img)
    
    # 이미지 면적의 0.1% 보다 작은 노이즈 제거
    min_contour_area = (w * h) * 0.001 
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    cv2.drawContours(clean_mask, large_contours, -1, 255, thickness=1)

    # ---  끊어진 선 강제 연결  ---
    kernel_size = int(diag * 0.005)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    connected_edges = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    
    if debug and debug_image is not None:
        # 연결된 엣지 확인용 (선택)
        debug_conn = cv2.resize(connected_edges, (0,0), fx=0.3, fy=0.3)
        # cv2.imshow("DEBUG_Connected_Edges", debug_conn)

    # --- 직선 검출 ---  
    # min_length: 대각선의 3% 이상 (너무 짧은 선 무시)
    dyn_min_len = int(diag * 0.03)
    # max_gap: 대각선의 10% (약 500px) -> 문양 전체를 건너뛸 수 있음
    dyn_max_gap = int(diag * 0.10)
    
    lines = cv2.HoughLinesP(connected_edges, 1, np.pi / 180, 
                            threshold=10,
                            minLineLength=dyn_min_len, 
                            maxLineGap=dyn_max_gap) 

    if lines is None:
        # print("[WARN] 직선을 찾지 못했습니다.")
        return 100, int(100 * aspect_ratio)

    # --- 가장 긴 선 찾기 ---
    line_data = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        line_data.append((length, x1, y1, x2, y2))

    line_data.sort(key=lambda x: x[0], reverse=True)

    max_len = line_data[0][0]
    
    # 1등 길이의 85% 이상 되는 선들만 평균
    valid_lengths = [x[0] for x in line_data if x[0] > max_len * 0.85]
    
    if not valid_lengths:
        estimated_h = max_len
    else:
        estimated_h = np.mean(valid_lengths)

    estimated_w = estimated_h / aspect_ratio
    
    if debug:
        # print(f"[Size Est] Max: {max_len:.1f}, Avg: {estimated_h:.1f}, GapUsed: {dyn_max_gap}")
        pass

    # --- 시각화 ---
    if debug and debug_image is not None:
        pass
        # vis = debug_image.copy()
        # mask_color = cv2.cvtColor(connected_edges, cv2.COLOR_GRAY2BGR)
        # vis = cv2.addWeighted(vis, 0.7, mask_color, 0.3, 0)

        # for i, ld in enumerate(line_data):
        #     length, x1, y1, x2, y2 = ld
        #     if length > max_len * 0.85:
        #         cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 5)
        #     elif i < 5: 
        #         cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # info_txt = f"Est H: {estimated_h:.1f}"
        # cv2.putText(vis, info_txt, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        # cv2.namedWindow("DEBUG_Estimate_Size", cv2.WINDOW_NORMAL)
        # h_vis, w_vis = vis.shape[:2]
        # if h_vis > 800:
        #     cv2.resizeWindow("DEBUG_Estimate_Size", w_vis//2, h_vis//2)
        # cv2.imshow("DEBUG_Estimate_Size", vis)
        # cv2.waitKey(0)
        # cv2.destroyWindow("DEBUG_Estimate_Size")
        # if cv2.getWindowProperty("DEBUG_Connected_Edges", 0) >= 0:
        #     cv2.destroyWindow("DEBUG_Connected_Edges")

    return int(estimated_w), int(estimated_h)


def debug_ght_heatmap(gray_det, edge_img, card_w, card_h, angle_step_deg=5):
    h, w = gray_det.shape[:2]
    if card_h > card_w: card_w, card_h = card_h, card_w

    # 1) Gradient & Phi
    gx = cv2.Sobel(gray_det, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_det, cv2.CV_32F, 0, 1, ksize=3)
    phi = np.arctan2(gy, gx)
    phi[phi < 0] += np.pi

    # 2) 파라미터
    R, r = card_w / 2.0, card_h / 2.0
    angle_step = np.deg2rad(angle_step_deg)
    n_phi = int(np.pi / angle_step)
    H = np.zeros((h, w), dtype=np.float32)

    # 3) Vectorized Voting 준비
    y_idxs, x_idxs = np.where(edge_img > 0)
    phis = phi[y_idxs, x_idxs]
    
    # v=Normal, w=Tangent
    cos_p, sin_p = np.cos(phis), np.sin(phis)
    v_x, v_y = cos_p[:, None], sin_p[:, None]
    w_x, w_y = -sin_p[:, None], cos_p[:, None]
    pts_x, pts_y = x_idxs[:, None], y_idxs[:, None]

    step_sampling = 4
    
    # 가설 생성 (긴 변 & 짧은 변 / 양방향)
    j_vals = np.arange(-int(R), int(R) + 1, step_sampling)[None, :]
    k_vals = np.arange(-int(r), int(r) + 1, step_sampling)[None, :]

    # 후보 좌표 계산 (4가지 케이스)
    c_list = []
    # Long edge hypothesis (+v, -v)
    c_list.append((pts_x + r * v_x + j_vals * w_x, pts_y + r * v_y + j_vals * w_y))
    c_list.append((pts_x - r * v_x + j_vals * w_x, pts_y - r * v_y + j_vals * w_y))
    # Short edge hypothesis (+v, -v)
    c_list.append((pts_x + R * v_x + k_vals * w_x, pts_y + R * v_y + k_vals * w_y))
    c_list.append((pts_x - R * v_x + k_vals * w_x, pts_y - R * v_y + k_vals * w_y))

    all_cx = np.concatenate([c[0].ravel() for c in c_list])
    all_cy = np.concatenate([c[1].ravel() for c in c_list])

    # 유효 좌표 필터링
    all_cx = np.rint(all_cx).astype(np.int32)
    all_cy = np.rint(all_cy).astype(np.int32)
    mask = (all_cx >= 0) & (all_cx < w) & (all_cy >= 0) & (all_cy < h)
    
    # 2D 투표 실행
    np.add.at(H, (all_cy[mask], all_cx[mask]), 1)

    # 4) 시각화 (Heatmap 생성)
    H_norm = cv2.normalize(H, None, 0, 255, cv2.NORM_MINMAX)
    H_img = H_norm.astype(np.uint8)
    
    # 보기 좋게 컬러맵 적용
    H_color = cv2.applyColorMap(H_img, cv2.COLORMAP_JET)
    
    # print(f"[DEBUG-GHT] Peak Vote: {H.max()}")
    return H_color

def trace_votes_for_centers(gray_det, edge_img, card_w, card_h, centers, angle_step_deg=2):
    h, w = gray_det.shape[:2]
    if card_h > card_w: card_w, card_h = card_h, card_w

    # 시각화용 이미지
    vis_map = cv2.cvtColor(gray_det, cv2.COLOR_GRAY2BGR)
    
    # 파라미터
    R_long = card_h / 2.0
    R_short = card_w / 2.0
    
    # Gradient
    gx = cv2.Sobel(gray_det, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_det, cv2.CV_32F, 0, 1, ksize=3)
    phi = np.arctan2(gy, gx)
    phi[phi < 0] += np.pi

    y_idxs, x_idxs = np.where(edge_img > 0)
    phis = phi[y_idxs, x_idxs]
    
    cos_p, sin_p = np.cos(phis), np.sin(phis)
    v_x, v_y = cos_p, sin_p  # Normal
    w_x, w_y = -sin_p, cos_p # Tangent
    
    # 밝기 방향 판별
    check_dist = 3
    pts_x, pts_y = x_idxs, y_idxs
    
    chk_x_pos = np.clip(pts_x + v_x * check_dist, 0, w-1).astype(int)
    chk_y_pos = np.clip(pts_y + v_y * check_dist, 0, h-1).astype(int)
    chk_x_neg = np.clip(pts_x - v_x * check_dist, 0, w-1).astype(int)
    chk_y_neg = np.clip(pts_y - v_y * check_dist, 0, h-1).astype(int)
    
    val_pos = gray_det[chk_y_pos, chk_x_pos]
    val_neg = gray_det[chk_y_neg, chk_x_neg]
    signs = np.where(val_pos > val_neg, 1.0, -1.0)

    step_sampling = 3
    j_vals = np.arange(-int(R_long), int(R_long) + 1, step_sampling)
    k_vals = np.arange(-int(R_short), int(R_short) + 1, step_sampling)

    # 각 센터별로 역추적
    for idx, (cx, cy) in enumerate(centers):
        # print(f"[Trace] Analyzing votes for Center #{idx}: ({cx}, {cy})")
        
        # 이 센터 근처에 투표한 픽셀들을 찾기 위한 마스크
        radius_tol = 5.0 
        # -- 가설 1: Long Edge (R_short 이동) --
        center_vec = np.array([cx, cy])
        
        # 모든 에지 점에 대해 벡터 연산
        # (Hypothesis 1 check)
        votes_long_x = pts_x[:, None] + R_short * v_x[:, None] * signs[:, None] + w_x[:, None] * j_vals[None, :]
        votes_long_y = pts_y[:, None] + R_short * v_y[:, None] * signs[:, None] + w_y[:, None] * j_vals[None, :]
        
        dist_long = np.hypot(votes_long_x - cx, votes_long_y - cy)
        # dist_long < radius_tol 인 점의 인덱스 (에지 점 인덱스)
        valid_long_mask = np.any(dist_long < radius_tol, axis=1)
        
        # -- 가설 2: Short Edge (R_long 이동) --
        votes_short_x = pts_x[:, None] + R_long * v_x[:, None] * signs[:, None] + w_x[:, None] * k_vals[None, :]
        votes_short_y = pts_y[:, None] + R_long * v_y[:, None] * signs[:, None] + w_y[:, None] * k_vals[None, :]
        
        dist_short = np.hypot(votes_short_x - cx, votes_short_y - cy)
        valid_short_mask = np.any(dist_short < radius_tol, axis=1)
        
        # 그리기
        # Long Edge 가설로 투표한 픽셀 -> 빨간색
        vis_local = vis_map.copy()
        
        vis_local[pts_y[valid_long_mask], pts_x[valid_long_mask]] = (0, 0, 255) 
        # Short Edge 가설로 투표한 픽셀 -> 파란색
        vis_local[pts_y[valid_short_mask], pts_x[valid_short_mask]] = (255, 0, 0)
        
        # 중심점 -> 노란색
        # cv2.circle(vis_local, (int(cx), int(cy)), 5, (0, 255, 255), -1)
        # cv2.putText(vis_local, f"Center #{idx}", (int(cx)+10, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # win_name = f"DEBUG_Trace_Center_{idx}"
        # cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(win_name, w//2, h//2)
        # cv2.imshow(win_name, vis_local)
    
    # print("[Trace] Press any key to close debug windows...")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def correct_card_orientation(corners_set):
    corrected_set = []
    
    for corners in corners_set:
        # corners 구조: [[[x,y]], [[x,y]], [[x,y]], [[x,y]]] (4, 1, 2)
        # 계산 편의를 위해 (4, 2) numpy array로 변환
        pts = np.array([p[0] for p in corners], dtype=np.float32)
        
        # TL(0) -> BL(1) 거리 (Height)
        height_approx = np.linalg.norm(pts[0] - pts[1])
        # TL(0) -> TR(3) 거리 (Width)
        width_approx = np.linalg.norm(pts[0] - pts[3])
        
        # 가로가 더 길면 90도 회전 (누운 카드)
        if width_approx > height_approx:
            new_pts = np.array([pts[1], pts[2], pts[3], pts[0]], dtype=np.float32)
        else:
            new_pts = pts
            
        new_corners = [[[int(p[0]), int(p[1])]] for p in new_pts]
        corrected_set.append(new_corners)
        
    return corrected_set