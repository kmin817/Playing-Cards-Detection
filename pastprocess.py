import cv2
import numpy as np
import os
import time

from src.ColorHelper import ColorHelper
from src.utils.DistanceHelper import DistanceHelper
# from src.utils import constants
# from src.utils import model_wrapper

CARD_RATIO = 1.45

# [방법 1] Otsu Binarization (Blob 방식) - 기본
# 배경이 깔끔할 때 가장 안정적이고 빠름
def get_thresh_otsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 노이즈 제거를 위한 블러
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Otsu 이진화
    _, thresh = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    kernel_open = np.ones((3, 3), np.uint8)
    # iterations=2 정도면 1~2픽셀 두께의 선도 확실하게 끊어집니다.
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # 모폴로지 연산 (구멍 메우기)
    kernel = np.ones((5, 5), np.uint8)    
    # Closing: 내부의 검은 무늬(숫자/모양)를 메워 흰색 덩어리로 만듦
    dial = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return dial

# [방법 2] Auto Canny (Edge 방식) - 백업/조명 대응
# 조명 반사(Glare)나 배경 질감이 복잡할 때 사용
def get_thresh_canny(img):
    """
    기존 Otsu 방식에서 Canny Edge 방식으로 변경.
    조명(Glare)이 강한 상황에서도 카드의 테두리 선은 살아있으므로 이를 검출합니다.
    """
    # 1. 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 블러 (노이즈 제거)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # 3. Canny 엣지 검출 (Threshold는 상황에 맞춰 조절 가능, 50/150은 일반적)
    edge = cv2.Canny(blur, 50, 150)
    
    # 4. 끊어진 엣지 연결 (Morphology Dilate)
    # 선을 약간 두껍게 만들어서 미세하게 끊긴 부분을 이어줍니다.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dial = cv2.dilate(edge, kernel, iterations=2)
    
    # (선택사항) 내부를 채우지 않아도 findContours(RETR_EXTERNAL)은 
    # 가장 바깥쪽 폐곡선을 잘 찾아내므로 이대로 리턴해도 됩니다.
    return dial

def find_corners_set(img, original, draw=True):
    # RETR_EXTERNAL: 가장 바깥쪽 외곽선만 검출
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    proper = sorted(contours, key=cv2.contourArea, reverse=True)

    four_corners_set = []
    for cnt in proper:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)

        x, y, w, h = cv2.boundingRect(cnt)

        # 면적 필터링 (카드가 아닌 작은 노이즈 무시)
        if area > 8000: 
            
            # [1단계] 1차 시도: 정밀 검사
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, closed=True)

            # [2단계] 실패 시 복구 로직: Convex Hull + 재시도
            if len(approx) != 4:
                hull = cv2.convexHull(cnt)
                hull_perimeter = cv2.arcLength(hull, closed=True)
                
                # Hull을 기준으로 다시 근사화 (오차 5%까지 허용)
                epsilon_hull = 0.05 * hull_perimeter 
                approx = cv2.approxPolyDP(hull, epsilon_hull, closed=True)
            
            num_corners = len(approx)

            if num_corners == 4:
                if draw:
                    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(original, "Found(Otsu/Canny)", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    
                pts = approx.reshape((4, 2))
                ordered_pts = np.zeros((4, 2), dtype=np.float32)

                sum_ = pts.sum(axis=1)
                ordered_pts[0] = pts[np.argmin(sum_)] 
                ordered_pts[2] = pts[np.argmax(sum_)] 

                diff_ = pts[:, 0] - pts[:, 1]
                ordered_pts[3] = pts[np.argmax(diff_)] 
                ordered_pts[1] = pts[np.argmin(diff_)] 

                height_approx = np.linalg.norm(ordered_pts[0] - ordered_pts[1])
                width_approx = np.linalg.norm(ordered_pts[0] - ordered_pts[3])

                if width_approx > height_approx:
                    ordered_pts = np.array([ordered_pts[1], ordered_pts[2], ordered_pts[3], ordered_pts[0]], dtype=np.float32)
                
                finalOrder = [[p.tolist()] for p in ordered_pts]
                four_corners_set.append(finalOrder)

            else:
                if draw:
                    cv2.drawContours(original, [approx], -1, (0, 0, 255), 3)
            
    return four_corners_set

# 카드 평탄화
def find_flatten_cards(img, set_of_corners, debug=False):
    width, height = 200, 300
    img_outputs = []

    for i, corners in enumerate(set_of_corners):
        top_left = corners[0][0]
        bottom_left = corners[1][0]
        bottom_right = corners[2][0]
        top_right = corners[3][0]

        # (미사용 변수지만 원본 로직 유지)
        _vertical_left = DistanceHelper.euclidean(top_left[0], top_left[1], bottom_left[0], bottom_left[1])
        _horizontal_top = DistanceHelper.euclidean(top_left[0], top_left[1], top_right[0], top_right[1])

        pts1 = np.float32([top_left, bottom_left, bottom_right, top_right])
        pts2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_output = cv2.warpPerspective(img, matrix, (width, height))
        img_outputs.append(img_output)

    return img_outputs

# 코너 스니핑
# 컨투어로 이 두 개를 분리해 각각 템플릿 매칭을 하기 위한 “원천 패치”를 만든다
def get_corner_snip(flattened_images: list):
    corner_images = []
    for img in flattened_images:
        # crop the image to where the corner might be
        crop = img[5:110, 1:38]

        # resize by a factor of 4
        crop = cv2.resize(crop, None, fx=4, fy=4)

        # threshold the corner
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

    cv2.drawContours(img, cnts_sort, -1, (0, 255, 0), 1)

    ranksuit = []
    _rank = None

    for i, cnt in enumerate(cnts_sort):
        x, y, w, h = cv2.boundingRect(cnt)
        x2, y2 = x + w, y + h
        crop = original[y:y2, x:x2]

        if i == 0:  # rank: 70 x 125
            crop = cv2.resize(crop, (70, 125), 0, 0)
            _rank = crop
        else:       # suit: 70 x 100
            crop = cv2.resize(crop, (70, 100), 0, 0)
            if debug and _rank is not None:
                r = cv2.resize(_rank, (70, 100), 0, 0)
                s = cv2.resize(crop, (70, 100), 0, 0)
                hcat = np.concatenate((r, s), axis=1)
                hcat = cv2.resize(hcat, (250, 200), 0, 0)
                # cv2.imshow("crop2", hcat)
                # cv2.waitKey(1)

        crop = ColorHelper.gray2bin(crop)
        crop = ColorHelper.reverse(crop)
        ranksuit.append(crop)

    return ranksuit


def template_matching(rank, suit, train_ranks, train_suits, show_plt=False) -> tuple[str, str]:
    """
    Finds best rank and suit matches for the query card. Differences
    the query card rank and suit images with the train rank and suit images.
    The best match is the rank or suit image that has the least difference.
    """
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
        # print(f"[DEBUG] Comparing rank with template '{train_rank.name}': diff = {rank_diff}")

        if rank_diff < best_rank_match_diff:
            best_rank_match_diff = rank_diff
            best_rank_match_name = train_rank.name
            best_rank_diff_img = diff_img.copy()

            if show_plt:
                vis = cv2.hconcat([
                    _to3c(rank),
                    _to3c(train_rank.img),
                    _to3c(best_rank_diff_img)
                ])
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
                vis = cv2.hconcat([
                    _to3c(suit),
                    _to3c(train_suit.img),
                    _to3c(best_suit_diff_img)
                ])
                # cv2.imshow("suit / template / diff(best-so-far)", vis)
                # cv2.waitKey(1)

    # Thresholding rule 유지
    # 최종 최소 차이가 랭크<2300, 슈트<1000이면 해당 이름 반환, 아니면 "Unknown" 처리
    rank_name = best_rank_match_name if best_rank_match_diff < 2300 else "Unknown"
    suit_name = best_suit_match_name if best_suit_match_diff < 1000 else "Unknown"

    # 질의/템플릿/Best Diff 순서로 이미지 표시
    if show_plt:
        # 마지막 뷰를 닫고 싶다면 호출부에서 cv2.destroyAllWindows() 수행
        cv2.waitKey(0)

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

