import cv2
import numpy as np
import statistics
import os
import math

class TemplateMatcher:
    def __init__(self):
        self.train_ranks = []
        self.train_suits = []
        self.rank_path = "assets/imgs/ranks"
        self.suit_path = "assets/imgs/suits"
        self.load_templates()

    def load_images_from_folder(self, folder, target_size):
        data_list = []
        if not os.path.exists(folder):
            return data_list
        for filename in os.listdir(folder):
            if filename.startswith('.'): continue
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                name = os.path.splitext(filename)[0]
                _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                img_processed = self.crop_content(img_bin)
                if img_processed is not None:
                    img_resized = cv2.resize(img_processed, target_size)
                    data_list.append({'name': name, 'img': img_resized})
        return data_list

    def crop_content(self, img_bin):
        contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        max_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_cnt)
        if w < 2 or h < 2: return None
        return img_bin[y:y+h, x:x+w]

    def load_templates(self):
        # print(">> 템플릿 로딩 중...")
        self.train_ranks = self.load_images_from_folder(self.rank_path, (70, 125))
        self.train_suits = self.load_images_from_folder(self.suit_path, (70, 100))
        # print(f"   Rank: {len(self.train_ranks)}개, Suit: {len(self.train_suits)}개")

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
        return result

    # 180도 회전 검사 로직 추가
    def match_with_rotation_search(self, roi_img):
        _, roi_thresh = cv2.threshold(roi_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # 1. 정방향(0도 기준) 탐색 수행
        res_normal = self._search_best_match(roi_thresh)
        
        # 2. 180도 회전 후 탐색 수행
        rotated_180 = cv2.rotate(roi_thresh, cv2.ROTATE_180)
        res_inverted = self._search_best_match(rotated_180)
        
        # 3. 둘 중 더 좋은 점수(Diff가 낮은 쪽) 선택
        # res 구조: (name, diff, type, angle, roi, tmpl)
        if res_inverted[1] < res_normal[1]: # 뒤집은 게 더 점수가 좋으면
            # print(f" [Info] 180도 회전 매칭 성공! ({res_inverted[0]})")
            best_res = res_inverted
            # 주의: 디버깅용 ROI도 뒤집힌 상태 그대로 반환 (그래야 시각화할 때 똑바로 보임)
        else:
            best_res = res_normal

        name, diff, mtype, angle, roi_out, tmpl_out = best_res

        # 'A'와 같은 특정 문자는 반드시 구멍(Hole)이 있어야 함
        if name in ['A', '0', '4', '6', '8', '9', 'Q', 'O', 'P', 'R', 'B', 'D']:
            # 현재 ROI에서 컨투어와 계층구조(Hierarchy) 다시 추출
            # 주의: 흰색이 글자라고 가정(fg 이미지). 배경이 검정.
            # 구멍을 찾으려면 외곽선 내부의 자식 컨투어가 있는지 봐야 함
            contours, hierarchy = cv2.findContours(roi_out, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            has_hole = False
            if hierarchy is not None:
                # hierarchy: [Next, Previous, First_Child, Parent]
                # First_Child(인덱스 2)가 -1이 아니면 자식(구멍)이 있다는 뜻
                for i in range(len(contours)):
                    if hierarchy[0][i][2] != -1:
                        has_hole = True
                        break
            
            # 구멍이 없는데 구멍이 필요한 글자로 인식됐다면? -> 점수 벌점 부여 또는 무시
            if not has_hole:
                diff += 5000  # 강제로 Diff를 높여서 Unknown 처리 되도록 유도
                # print(f" [Filter] {name} detected but no hole found -> Ignored")
        
        # 임계값 판정
        threshold = 2200 
        is_unknown = False
        if diff > threshold:
            is_unknown = True
            
        return name, diff, mtype, angle, roi_out, tmpl_out, is_unknown

    def _search_best_match(self, src_img):
        """
        주어진 이미지(src_img)에 대해 -45~+45도 회전을 하며 최적의 템플릿을 찾음
        """
        best_diff = 999999
        best_name = "Unknown"
        best_type = "Unknown"
        best_tmpl_img = None
        best_roi_img = None
        best_found_angle = 0
        
        # 검색 범위: -45 ~ 45
        for angle in range(-45, 46, 5):
            rotated_roi = self.rotate_image(src_img, angle)
            processed_roi = self.crop_content(rotated_roi)
            if processed_roi is None: continue

            # Rank Matching
            roi_rank = cv2.resize(processed_roi, (70, 125))
            for train in self.train_ranks:
                diff = int(np.sum(cv2.absdiff(roi_rank, train['img'])) / 255)
                if diff < best_diff:
                    best_diff = diff
                    best_name = train['name']
                    best_type = "Rank"
                    best_tmpl_img = train['img']
                    best_roi_img = roi_rank
                    best_found_angle = angle

            # Suit Matching
            roi_suit = cv2.resize(processed_roi, (70, 100))
            for train in self.train_suits:
                diff = int(np.sum(cv2.absdiff(roi_suit, train['img'])) / 255)
                if diff < best_diff:
                    best_diff = diff
                    best_name = train['name']
                    best_type = "Suit"
                    best_tmpl_img = train['img']
                    best_roi_img = roi_suit
                    best_found_angle = angle
        
        return (best_name, best_diff, best_type, best_found_angle, best_roi_img, best_tmpl_img)

def deskew_roi(roi_img):
    h, w = roi_img.shape[:2]
    contours, _ = cv2.findContours(roi_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return roi_img, 0
    max_cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_cnt)
    (cx, cy), (rect_w, rect_h), angle = rect
    
    if rect_w > rect_h: major, minor = rect_w, rect_h
    else: major, minor = rect_h, rect_w
    
    ratio = major / (minor + 1e-5)
    rotation_angle = angle
    
    if ratio > 1.2:
        if rect_w > rect_h: rotation_angle += 90
    else:
        if abs(rotation_angle) > 30: rotation_angle = 0 

    while rotation_angle > 45: rotation_angle -= 90
    while rotation_angle < -45: rotation_angle += 90

    M = cv2.getRotationMatrix2D((w // 2, h // 2), rotation_angle, 1.0)
    rotated = cv2.warpAffine(roi_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated, rotation_angle

def get_clean_card_info(rank_name, suit_name):
    """
    파일명에서 불필요한 접미사를 제거하고, 정렬을 위한 우선순위 값을 반환합니다.
    출력 형식: [Suit약어][Rank] (예: S5, H10, DA) -> 슬라이드 예시 기준
    """
    # 1. Rank 이름 정규화 (5_bold -> 5, Q2 -> Q)
    rank_clean = rank_name.split('_')[0]  # '_' 뒤 제거
    # 숫자가 아닌 문자(Q, K 등) 뒤에 붙은 숫자 제거 (Q2 -> Q)
    if not rank_clean.isdigit() and len(rank_clean) > 1:
        rank_clean = rank_clean.rstrip('0123456789')
    
    # '0' 또는 '10' 처리
    if rank_clean == '0': rank_clean = '10'

    # 2. Suit 이름 정규화 (Spades2 -> S)
    suit_map = {
        'Clubs': 'C', 'Diamonds': 'D', 'Hearts': 'H', 'Spades': 'S'
    }
    # 파일명에 포함된 키워드로 매핑 (예: "Spades2" -> "Spades" 포함 -> "S")
    suit_clean = '?'
    for key, val in suit_map.items():
        if key in suit_name:
            suit_clean = val
            break
            
    # 3. 정렬 우선순위 설정
    # Rank 순서: 2~9, 10, J, Q, K, A
    rank_order = {
        'A': 1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, 
        'J':11, 'Q':12, 'K':13
    }
    # Suit 순서: C < D < H < S
    suit_order = {'C': 0, 'D': 1, 'H': 2, 'S': 3, '?': 4}

    rank_val = rank_order.get(rank_clean, 0)
    suit_val = suit_order.get(suit_clean, 4)

    return {
        'rank_display': rank_clean,
        'suit_display': suit_clean,
        'rank_val': rank_val,
        'suit_val': suit_val,
        'full_str': f"{suit_clean}{rank_clean}" # 예: S5
    }

def visualize_pairing(img, detections):
    ranks = [d for d in detections if d['type'] == 'Rank']
    suits = [d for d in detections if d['type'] == 'Suit']
    
    detected_cards = []
    used_suit_indices = set()
    
    
    for rank in ranks:
        rx, ry = rank['center']
        rw, rh = rank['box'][2], rank['box'][3]
        
        search_radius = rh * 1.7 
        best_suit = None
        best_suit_idx = -1
        min_dist = 99999
        
        for i, suit in enumerate(suits):
            if i in used_suit_indices: continue

            sw, sh = suit['box'][2], suit['box'][3]
            if sh > rh * 1.8:
                continue
            
            sx, sy = suit['center']
            dist = math.sqrt((rx - sx)**2 + (ry - sy)**2)
            
            if dist > search_radius: continue
            if dist < min_dist:
                min_dist = dist
                best_suit = suit
                best_suit_idx = i
        
        if best_suit:
            # 짝을 찾음 -> 정보 정제
            info = get_clean_card_info(rank['name'], best_suit['name'])
            
            # 중복 체크를 위해 리스트에 임시 저장 (좌표는 시각화용)
            card_obj = {
                'info': info,
                'rank_center': (rx, ry),
                'suit_center': best_suit['center']
            }
            detected_cards.append(card_obj)
            used_suit_indices.add(best_suit_idx)
        else:
            # 짝을 못 찾음 -> '?' 처리
            cv2.putText(img, "?", (int(rx), int(ry)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 1. 중복 제거 (Rank와 Suit가 모두 같으면 제거)
    unique_cards = {}
    for card in detected_cards:
        key = card['info']['full_str']

        unique_cards[key] = card 
    
    final_list = list(unique_cards.values())

    # 2. 정렬 (1순위: Rank, 2순위: Suit)
    final_list.sort(key=lambda x: (x['info']['rank_val'], x['info']['suit_val']))

    # 3. 결과 텍스트 생성 및 시각화
    result_strings = []
    # print(" >>> 정렬된 카드 목록:")
    
    for card in final_list:
        info = card['info']
        rx, ry = card['rank_center']
        sx, sy = card['suit_center']
        
        # 1. 랭크와 슈트를 잇는 선 그리기 (연결 관계 확인용)
        cv2.line(img, (int(rx), int(ry)), (int(sx), int(sy)), (0, 255, 0), 2)
        
        # 2. 텍스트 위치 계산 
        # (Rank 심볼이 보통 카드 상단에 있으므로 rx, ry를 기준으로 잡음)
        text_x = int(rx) - 30
        text_y = int(ry) - 30
        
        # 화면 위쪽 짤림 방지
        if text_y < 50: 
            text_y = int(ry) + 90

        # 출력할 텍스트 (예: "S5", "HK")
        display_text = info['full_str']
        
        # 3. [크고 굵은 텍스트 그리기]
        # (1) 검은색 테두리 (가독성 확보, 두께 10)
        cv2.putText(img, display_text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 10, cv2.LINE_AA)
        
        # (2) 노란색/하늘색 글씨 (두께 4)
        cv2.putText(img, display_text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 4, cv2.LINE_AA)
        
        result_strings.append(info['full_str'])

    output_str = " ".join(result_strings)
    # print(f" [결과] {output_str}")

    return result_strings

def run_overlap_pipeline(img_path):
    src = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if src is None: 
        # print(f"[Error] 이미지를 불러올 수 없습니다: {img_path}")
        return

    matcher = TemplateMatcher()

    blur = cv2.GaussianBlur(src, (11, 11), 0)
    bin_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 25)
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    fg = cv2.bitwise_not(bin_img)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg)
    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0: return
    
    # --- 동적 임계값 설정 (Hybrid Threshold) ---
    img_area = src.shape[0] * src.shape[1]
    valid_areas = [a for a in areas if a > 200 and a < img_area * 0.5]
    
    if valid_areas:
        median_area = statistics.median(valid_areas)
        if median_area < 1000:
            min_area = median_area * 3.0
            max_area = median_area * 60.0
        else: 
            min_area = median_area * 0.2
            max_area = median_area * 5.0

        min_area = max(min_area, 500)
        max_area = min(max_area, 30000)
        # print(f"Median: {median_area:.1f} -> Min: {min_area:.1f}, Max: {max_area:.1f}")
    else:
        min_area, max_area = 500, 30000

    detected_objects = []

    # print("[분석 시작]")

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        
        if area < min_area or area > max_area: continue
        if w/h > 6.0 or w/h < 0.4: continue 

        pad = 2
        roi_raw = fg[max(0, y-pad):min(src.shape[0], y+h+pad), max(0, x-pad):min(src.shape[1], x+w+pad)]
        contours, _ = cv2.findContours(roi_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        is_ignored = False
        if len(contours) > 0:
            max_cnt = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(max_cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area) / hull_area
                if solidity < 0.3: is_ignored = True
        
        if is_ignored:
            cv2.rectangle(dst, (x, y), (x+w, y+h), (255, 0, 0), 1)
            continue

        pad = 15 
        roi = fg[max(0, y-pad):min(src.shape[0], y+h+pad), max(0, x-pad):min(src.shape[1], x+w+pad)]
        deskewed_roi, base_angle = deskew_roi(roi)
        
        name, diff, mtype, fine_angle, roi_used, tmpl_used, is_unknown = matcher.match_with_rotation_search(deskewed_roi)
        
        # Visual Debug
        if roi_used is not None and tmpl_used is not None:
            # 1. 메인 컨텍스트 표시
            temp_dst = dst.copy()
            cv2.rectangle(temp_dst, (x, y), (x+w, y+h), (255, 0, 255), 3)
            # cv2.namedWindow("Current Context", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("Current Context", 1200, 800)
            # cv2.imshow("Current Context", temp_dst)

            # 2. 매칭 상세 창 표시
            diff_img = cv2.absdiff(roi_used, tmpl_used)
            diff_vis = cv2.multiply(diff_img, 3.0) 
            vis_row = cv2.hconcat([
                cv2.cvtColor(roi_used, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(tmpl_used, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(diff_vis, cv2.COLOR_GRAY2BGR)
            ])
            
            # 정보 텍스트 (Diff 및 회전 여부 확인 가능)
            info_text = f"Match:{name}" if not is_unknown else f"Unk:{name}"
            cv2.putText(vis_row, info_text, (75, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(vis_row, f"Diff:{diff}", (145, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            # cv2.namedWindow("Debug Match", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("Debug Match", 1200, 800)
            # cv2.imshow("Debug Match", vis_row)
            
            key = cv2.waitKey(0)
            if key == 27: break

        if not is_unknown:
            detected_objects.append({
                'name': name,
                'type': mtype,
                'center': (cx, cy),
                'box': (x, y, w, h)
            })
            color = (0, 255, 0) if mtype == "Rank" else (0, 200, 255)
            cv2.rectangle(dst, (x, y), (x+w, y+h), color, 2)
            cv2.putText(dst, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.rectangle(dst, (x, y), (x+w, y+h), (100, 100, 100), 2)
            # cv2.putText(dst, f"Unk:{diff}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    final_results = visualize_pairing(dst, detected_objects)
    # print("\n >>> 최종 인식 결과:\n " + ", ".join(final_results) + "\n")
    filtered_results = [code for code in final_results if '?' not in code]
    print(" ".join(filtered_results))

    # cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Result", 1200, 800)
    # cv2.imshow("Result", dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    run_overlap_pipeline("testImgs/card1.jpg")
