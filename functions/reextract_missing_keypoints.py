import cv2, json  # OpenCV와 JSON 처리 라이브러리
import numpy as np  # 수치 연산용 numpy
from pathlib import Path  # 경로 처리를 위한 Path
from tqdm import tqdm  # 진행 상황 표시용 tqdm
from mmpose.apis import inference_topdown  # 포즈 추정 실행 함수
from mmpose.structures import merge_data_samples, split_instances  # 추론 결과 처리 함수

def to_py(obj):
    """넘파이 객체를 JSON 직렬화 가능한 타입으로 변환"""
    import numpy as _np
    if isinstance(obj, _np.ndarray): return obj.tolist()
    if isinstance(obj, (_np.floating,)): return float(obj)
    if isinstance(obj, (_np.integer,)):  return int(obj)
    if isinstance(obj, dict):  return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_py(v) for v in obj]
    return obj

def reextract_missing_keypoints(
    file_name: str,              # 비디오 파일명
    frame_dir: str,              # 프레임 디렉토리
    json_dir: str,               # JSON 저장 디렉토리
    n_extracted_frames: int,     # 총 추출된 프레임 수
    pose_estimator,              # 초기화된 Sapiens 포즈 추정 모델
) -> int:
    """
    누락된 프레임만 Sapiens로 재추출 (bbox는 인접 JSON에서 재활용)

    Returns:
        int: 최종 JSON 개수
    """
    frame_dir, json_dir = Path(frame_dir), Path(json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)

    # 기대되는 모든 프레임 번호 집합
    expected = {f"{i:06d}" for i in range(n_extracted_frames)}
    # 이미 존재하는 JSON 번호 집합
    existing = {p.stem for p in json_dir.glob("*.json")}
    # 누락된 프레임 번호 계산
    missing = sorted(expected - existing)

    if not missing:
        print(f"[INFO] {file_name}: 누락된 프레임 없음")
        return len(existing)  # 최종 JSON 개수 반환

    for fidx_str in tqdm(missing, desc=f"{file_name} (re-infer)", unit="frame"):
        fidx = int(fidx_str)
        fpath = frame_dir / f"{fidx:06d}.jpg"
        jpath = json_dir / f"{fidx:06d}.json"

        if not fpath.exists() or jpath.exists():
            continue

        img_bgr = cv2.imread(str(fpath))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # -------------------------------
        # bbox 재활용 (좌우 인접 JSON 탐색)
        # -------------------------------
        neighbor = None
        off = 1
        while True:
            left = json_dir / f"{fidx-off:06d}.json"
            right = json_dir / f"{fidx+off:06d}.json"
            if left.exists():
                neighbor = left
                break
            if right.exists():
                neighbor = right
                break
            if (fidx-off) < 0 and (fidx+off) >= n_extracted_frames:
                break
            off += 1

        if neighbor is None:
            continue

        with open(neighbor, "r", encoding="utf-8") as f:
            nb = json.load(f)
        if not nb.get("instance_info"):
            continue

        bbox = np.array(nb["instance_info"][0]["bbox"], dtype=np.float32).reshape(1,4)

        # -------------------------------
        # Sapiens 포즈 재추론
        # -------------------------------
        results = inference_topdown(pose_estimator, img_rgb, bbox)
        data_sample = merge_data_samples(results)
        inst = data_sample.get("pred_instances", None)
        if inst is None:
            continue
        inst_list = split_instances(inst)

        # -------------------------------
        # JSON 저장
        # -------------------------------
        payload = dict(
            frame_index=fidx,
            video_name=file_name,
            meta_info=pose_estimator.dataset_meta,
            instance_info=inst_list,
            source="reextract"
        )
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(to_py(payload), f, ensure_ascii=False, indent=2)

    # 최종 JSON 개수 다시 세서 반환
    final_json_count = len(list(json_dir.glob("*.json")))
    print(f"[INFO] {file_name}: 최종 JSON 개수 {final_json_count}")
    return final_json_count
