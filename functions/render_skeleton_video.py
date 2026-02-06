import cv2, json
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
from functions.constants_skeleton.registry import load_skeleton_constants


def render_skeleton_video(
    frame_dir: str,
    json_dir: str,
    out_mp4: str,
    fps: int = 30,
    kp_radius: int = 4,
    line_thickness: int = 2,
    model_type: str = "coco17",     # ✅ constants_skeleton에서 모델 타입 지정
    flip_horizontal: bool = True   # ✅ 선택적 반전
):
    """
    프레임 + keypoints JSON → skeleton overlay mp4 생성
    - model_type: 'coco17', 'yolo12' 등 constants_skeleton에서 로드
    - flip_horizontal: 좌우 반전 여부 (기본 False)
    """

    frame_files = sorted(Path(frame_dir).glob("*.jpg"))
    if not frame_files:
        print(f"[WARN] No frames found in {frame_dir}")
        return

    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # ✅ 모델에 맞는 Skeleton/Color 상수 불러오기
    const = load_skeleton_constants(model_type)
    COLOR_SK = const.COLOR_SK
    COLOR_L = const.COLOR_L
    COLOR_R = const.COLOR_R
    COLOR_NEUTRAL = const.COLOR_NEUTRAL
    LEFT_POINTS = const.LEFT_POINTS
    RIGHT_POINTS = const.RIGHT_POINTS
    EXCLUDE_POINTS = getattr(const, "EXCLUDE_POINTS", [])
    SKELETON_LINKS = getattr(const, "SKELETON_LINKS", [])

    # 해상도 확인
    sample = cv2.imread(str(frame_files[0]))
    h, w = sample.shape[:2]
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for frame_path in tqdm(frame_files, total=len(frame_files),
                           desc=f"{Path(frame_dir).name}", unit="frame"):
        frame = cv2.imread(str(frame_path))
        json_path = Path(json_dir) / (frame_path.stem + ".json")
        frame_number_str = frame_path.stem

        if not json_path.exists():
            writer.write(frame)
            continue

        # JSON 로드
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "instance_info" not in data or len(data["instance_info"]) == 0:
            writer.write(frame)
            continue

        # 기존
        # inst = data["instance_info"][0]
        # kpts = np.array(inst["keypoints"])

        # 수정 버전 ✅ 모든 skeleton 그리기
        for person in data["instance_info"]:
            kpts = np.array(person["keypoints"])

            # Skeleton 라인
            for i, j in SKELETON_LINKS:
                if i >= len(kpts) or j >= len(kpts):
                    continue
                if i in EXCLUDE_POINTS or j in EXCLUDE_POINTS:
                    continue
                pt1, pt2 = tuple(map(int, kpts[i])), tuple(map(int, kpts[j]))
                cv2.line(frame, pt1, pt2, COLOR_SK, line_thickness)

            # Keypoints 점
            for idx, (x, y) in enumerate(kpts):
                if idx in EXCLUDE_POINTS or x <= 0 or y <= 0:
                    continue
                if idx in LEFT_POINTS:
                    color = COLOR_L
                elif idx in RIGHT_POINTS:
                    color = COLOR_R
                else:
                    color = COLOR_NEUTRAL
                cv2.circle(frame, (int(x), int(y)), kp_radius, color, -1)


        # # 안내 문구
        legend_text = f"L: Blue   |   R: Red   |   Frame: {frame_number_str}"
        cv2.putText(frame, legend_text, (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # 좌우 반전 (선택)
        if flip_horizontal:
            frame = cv2.flip(frame, 1)

        writer.write(frame)

    writer.release()
    print(f"✅ Skeleton overlay 완료 → {out_mp4}")


def generate_133kpt_skeleton_video(
    frame_dir: str, 
    kpt_dir: str, 
    output_path: str, 
    show_hands: bool = False, 
    conf_threshold: float = 0.0007
):
    """
    Sapiens Keypoint 결과를 기반으로 의미상 좌우가 구분된 스켈레톤 영상을 생성합니다.
    """
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

    # 1. 시각화 대상 및 의미적 좌우 인덱스 정의
    target_indices = set(range(5, 24)) # 기본 몸통
    if show_hands:
        hand_indices = [91, 92, 94, 97, 109, 112, 113, 115, 118, 130]
        target_indices.update(hand_indices)

    # COCO Wholebody 기준 의미론적 좌우 (얼굴/몸통 홀수:좌, 짝수:우 / 손 91-111:좌, 112-132:우)
    left_indices = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, *range(23, 91, 2), *range(91, 112)}
    right_indices = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, *range(24, 91, 2), *range(112, 133)}

    # 2. 색상 설정 (BGR)
    COLOR_SKELETON = (50, 50, 50)   # 뼈대 고정: 짙은 회색
    COLOR_RIGHT = (0, 0, 255)       # 오른쪽: Red
    COLOR_LEFT = (255, 0, 0)        # 왼쪽: Blue
    COLOR_CENTER = (0, 255, 0)      # 중앙: Green
    COLOR_BBOX = (0, 255, 0)
    COLOR_TEXT = (255, 255, 255)

    # 3. 비디오 초기화
    with open(json_files[0], 'r') as f:
        first_data = json.load(f)
        skeleton_links = first_data.get('meta_info', {}).get('skeleton_links', [])

    first_frame_name = first_data.get('file_name', json_files[0].stem + ".jpg")
    img = cv2.imread(str(frame_path / first_frame_name))
    h, w = img.shape[:2]
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

    # 4. 프레임 처리
    for json_file in tqdm(json_files, desc="Rendering Video"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        fname = data.get('file_name', json_file.stem + ".jpg")
        frame = cv2.imread(str(frame_path / fname))
        if frame is None: continue

        for inst in data.get('instance_info', []):
            if inst.get('score', 1.0) <= conf_threshold: continue
            
            kpts = np.array(inst['keypoints'])
            coords = kpts[:, :2]
            scores = inst.get('keypoint_scores', kpts[:, 2] if kpts.shape[1] >= 3 else np.ones(len(coords)))

            # --- [Step 1] Skeleton 그리기 (고정 색상) ---
            for u, v in skeleton_links:
                if u >= len(coords) or v >= len(coords): continue
                if u <= 4 or v <= 4: continue # 얼굴 제외
                if scores[u] > conf_threshold and scores[v] > conf_threshold:
                    pt1 = (int(coords[u][0]), int(coords[u][1]))
                    pt2 = (int(coords[v][0]), int(coords[v][1]))
                    cv2.line(frame, pt1, pt2, COLOR_SKELETON, 1, cv2.LINE_AA)

            # --- [Step 2] Keypoints 그리기 (의미상 좌우 색상) ---
            for i, kp in enumerate(coords):
                if i <= 4: continue # 얼굴 중심부 제외
                if i in target_indices and scores[i] > conf_threshold:
                    x, y = int(kp[0]), int(kp[1])
                    
                    # 인덱스 기반 색상 선택
                    if i in right_indices:
                        color = COLOR_RIGHT
                    elif i in left_indices:
                        color = COLOR_LEFT
                    else:
                        color = COLOR_CENTER

                    cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)
                    cv2.putText(frame, str(i), (x + 3, y - 3), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_TEXT, 1, cv2.LINE_AA)

            # BBox 그리기 (선택 사항)
            bbox = inst.get('bbox')
            if bbox:
                x1, y1, x2, y2 = map(int, np.array(bbox).flatten())
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX, 1)

        out.write(frame)

    out.release()
    print(f"\n✅ 시각화 완료: {save_path}")