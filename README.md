# Playing-Cards-Detection

- OpenCV 기본 라이브러리를 활용한 카드 검출 및 분류 프로젝트

# Sort Playing Cards

OpenCV 기반으로 트럼프 카드(playing card)의 rank(A, 2-10, J, Q, K)와 suit(♣, ♦, ♥, ♠)를 인식한 뒤, 다음의 정렬 규칙에 따라 정렬 결과를 출력하세요.

- 정렬 규칙
  - Rank (A→K) 순서로 우선 정렬 - (오름차순): A < 2 < 3 < … < 10 < J < Q < K
  - 같은 rank인 경우 다음의 suit 순서(♣→♦→♥→♠)로 정렬 - (오름차순): ♣ < ♦ < ♥ < ♠
  - 출력시 각 suit는 다음의 알파벳으로 치환하여 출력 **(대문자로 출력)**
    - ♣ → C
    - ♦ → D
    - ♥ → H
    - ♠ → S
  - 출력은 suit 다음에 rank를 붙여서 문자열로 출력
    - 예시) CA, D3, S10, HK

# 입출력 예시

## 입력 예시

```bash
python main.py --input path/to/image.jpg
```

- input의 인자로 이미지의 파일명(현재 폴더가 아닌 경우 경로 포함)을 명령행 인자로 입력
- 자동 채점 시 script 상에서 project 폴더 밖에 있는 다른 경로에 있는 이미지를 사용할 예정

## 출력 예시

```bash
CA S2 C9 H10 DQ HQ HK SK
```

- 인식된 모든 카드들에 대해서 각 카드들은 공백으로 구분하여 한 줄로 출력
- 각 카드는 suit에 대항하는 알파벳을 먼저 출력하고, 그 뒤에 rank를 공백 없이 붙여서 출력
- ** 텍스트 출력 외에 cv2.imshow 등으로 이미지를 띄우지 마세요.**
  - 결과 이미지를 띄우고 waitkey 함수 등으로 대기하는 경우 수행 시간이 초과되어 0점 처리 될 수 있습니다.
  - main.py의 예시 코드에 있는 imshow를 비롯한 예시 코드는 지우고 작성해주세요.
- 최초 실행 후 채점 서버에서 3분 이상 소요되는 경우 오답 처리됩니다.

# 사용 패키지

- NumPy 2.2.6
- opencv-python 4.12.0.88
- opencv-contrib-python 4.12.0.88
