import cv2, json
import numpy as np
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
