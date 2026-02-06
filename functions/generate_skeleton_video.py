import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def generate_skeleton_video(
    frame_dir: str, 
    kpt_dir: str, 
    output_path: str, 
    show_hands: bool = False, 
    conf_threshold: float = 0.0007
):
    frame_path = Path(frame_dir)
    json_path = Path(kpt_dir)
    save_path = Path(output_path)

    if not json_path.exists():
        print(f"❌ JSON 경로를 찾을 수 없습니다: {json_path}")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    json_files = sorted(list(json_path.glob("*.json")))
    if not json_files:
        print("❌ 처리할 JSON 파일이 없습니다.")
        return

    # 1. 색상 설정 (BGR)
    COLOR_SKELETON = (50, 50, 50)   # 뼈대 고정: 짙은 회색
    COLOR_RIGHT = (0, 0, 255)       # 오른쪽: Red
    COLOR_LEFT = (255, 0, 0)        # 왼쪽: Blue
    COLOR_BBOX = (0, 255, 0)        # ID 박스: Green
    COLOR_TEXT = (255, 255, 255)

    # 2. 첫 번째 JSON에서 메타데이터(Skeleton 구조 및 좌우 매핑) 로드
    with open(json_files[0], 'r') as f:
        first_data = json.load(f)
        meta = first_data.get('meta_info', {})
        skeleton_links = meta.get('skeleton_links', [])
        kpt_name2id = meta.get('keypoint_name2id', {})

    # 의미상 좌우 인덱스 자동 분류 로직
    left_indices = set()
    right_indices = set()
    for name, idx in kpt_name2id.items():
        if 'left' in name:
            left_indices.add(idx)
        elif 'right' in name:
            right_indices.add(idx)

    # 시각화 대상 인덱스 설정 (5번 어깨부터 몸통 위주)
    target_indices = set(range(5, 23)) 
    if show_hands:
        # 손 관련 주요 관절 자동 추가 (name2id에서 검색)
        hand_keywords = ['hand', 'finger', 'thumb']
        for name, idx in kpt_name2id.items():
            if any(k in name for k in hand_keywords):
                target_indices.add(idx)

    # 3. 비디오 설정
    first_frame_name = first_data.get('file_name', json_files[0].stem + ".jpg")
    img = cv2.imread(str(frame_path / first_frame_name))
    h, w = img.shape[:2]
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

    # 4. 프레임 처리 루프
    for json_file in tqdm(json_files, desc="Rendering Video"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        fname = data.get('file_name', json_file.stem + ".jpg")
        frame = cv2.imread(str(frame_path / fname))
        if frame is None: continue

        for inst in data.get('instance_info', []):
            if inst.get('score', 1.0) <= conf_threshold: continue
            
            # 데이터 추출 (배열 처리)
            kpts_raw = np.array(inst['keypoints'])
            coords = kpts_raw[:, :2]
            
            # 신뢰도 점수
            if 'keypoint_scores' in inst:
                scores = np.array(inst['keypoint_scores'])
            else:
                scores = kpts_raw[:, 2] if kpts_raw.shape[1] >= 3 else np.ones(len(coords))

            obj_id = inst.get('instance_id', inst.get('id', '?'))

            # --- [Step 1] Skeleton 그리기 (JSON 링크 활용) ---
            for u, v in skeleton_links:
                if u >= len(coords) or v >= len(coords): continue
                
                # 🔴 [수정됨] 연결되는 두 점(u, v)이 모두 '시각화 대상(target_indices)'에 포함될 때만 선을 그립니다.
                # 이렇게 하면 show_hands=False일 때 손가락 관절 인덱스가 target_indices에 없으므로 선도 그려지지 않습니다.
                if u not in target_indices or v not in target_indices:
                    continue

                # 얼굴(0~4) 제외 (target_indices에 0~4가 없다면 위 조건에서 이미 걸러지지만, 명시적으로 유지해도 됩니다)
                # if u <= 4 or v <= 4: continue 
                
                if scores[u] > conf_threshold and scores[v] > conf_threshold:
                    pt1 = (int(coords[u][0]), int(coords[u][1]))
                    pt2 = (int(coords[v][0]), int(coords[v][1]))
                    cv2.line(frame, pt1, pt2, COLOR_SKELETON, 1, cv2.LINE_AA)

            # --- [Step 2] Keypoints 그리기 (의미론적 색상) ---
            for i, kp in enumerate(coords):
                if i not in target_indices: continue # 🔴 점 그리기 전에도 체크
                
                if scores[i] > conf_threshold:
                    # 인덱스 기반 색상 선택
                    if i in right_indices:
                        color = COLOR_RIGHT
                    elif i in left_indices:
                        color = COLOR_LEFT
                    else:
                        color = (0, 255, 0) # 중앙부 등

                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, color, -1, cv2.LINE_AA)
                    
                    # 텍스트가 너무 많으면 지저분하므로 손은 제외하고 싶다면 아래 조건 추가 가능
                    # if show_hands or i < 23: 
                    cv2.putText(frame, str(i), (int(kp[0]) + 3, int(kp[1]) - 3), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_TEXT, 1, cv2.LINE_AA)

            # --- [Step 3] ID 및 BBox 표시 ---
            bbox = inst.get('bbox')
            if bbox:
                b = np.array(bbox).flatten()
                x1, y1, x2, y2 = map(int, b[:4])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX, 1)
                
                label = f"ID: {obj_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw + 10, y1), COLOR_BBOX, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"\n✅ 비디오 생성 완료: {save_path}")