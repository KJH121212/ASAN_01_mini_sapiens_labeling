# 영상에서 Frame을 추출하는 코드

import cv2  # OpenCV 불러오기
from pathlib import Path  # 경로 처리
import shutil  # 폴더 삭제용
from tqdm import tqdm  # 진행 상황 표시

def extract_frames(video_path: str, frame_dir: str, target_short: int = 720, jpeg_quality: int = 80) -> int:
    """영상에서 프레임 추출 (720p 다운샘플링 저장)"""
    video_path, frame_dir = Path(video_path), Path(frame_dir)
    if frame_dir.exists():
        shutil.rmtree(frame_dir)  # 기존 폴더 삭제
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))  # 비디오 열기
    if not cap.isOpened():
        return 0  # 열기 실패 시 0 반환

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 원본 가로 크기
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 원본 세로 크기
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 총 프레임 수

    # 720p 기준으로 리사이즈 스케일 계산
    scale = target_short / w if w <= h else target_short / h
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    count = 0
    # tqdm으로 진행상황 표시
    for idx in tqdm(range(n_frames), total=n_frames, desc="Extracting Frames"):
        ret, frame = cap.read()
        if not ret:
            break
        # 프레임 리사이즈
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # 프레임 저장 (000000.jpg 형식)
        out_path = frame_dir / f"{idx:06d}.jpg"
        cv2.imwrite(str(out_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        count += 1

    cap.release()
    return count
