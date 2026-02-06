import cv2, json, shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_estimator, inference_topdown
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples, split_instances


# numpy → JSON 직렬화 변환 함수
def to_py(obj):
    """numpy 객체를 JSON 직렬화 가능한 Python 객체로 변환"""
    import numpy as _np
    if isinstance(obj, _np.ndarray): 
        return obj.tolist()                       # numpy array → list
    if isinstance(obj, (_np.floating,)): 
        return float(obj)                         # numpy float → float
    if isinstance(obj, (_np.integer,)):  
        return int(obj)                           # numpy int → int
    if isinstance(obj, dict):  
        return {k: to_py(v) for k, v in obj.items()}  # dict 내부 재귀 처리
    if isinstance(obj, (list, tuple)): 
        return [to_py(v) for v in obj]            # list/tuple 내부 재귀 변환
    return obj                                    # 기본 타입 그대로 반환


# Keypoints 추출 (Batch 버전)
def extract_keypoints(frame_dir: str, json_dir: str,
                      det_cfg: str, det_ckpt: str,
                      pose_cfg: str, pose_ckpt: str,
                      device: str = "cuda:0",
                      batch_size: int = 8) -> int:
    """
    주어진 프레임 디렉토리에서 사람 감지 + 포즈 추정 후
    각 프레임별 JSON 파일로 keypoints를 저장합니다.

    Args:
        frame_dir (str): 프레임 이미지(.jpg) 폴더 경로
        json_dir (str): JSON 결과를 저장할 폴더 경로
        det_cfg (str): Detector 설정 파일 경로 (mmdet config)
        det_ckpt (str): Detector checkpoint 파일 경로
        pose_cfg (str): Pose estimator 설정 파일 경로 (mmpose config)
        pose_ckpt (str): Pose estimator checkpoint 파일 경로
        device (str): 실행 장치 ("cuda:0" or "cpu")
        batch_size (int): Batch 단위로 처리할 프레임 수

    Returns:
        int: 생성된 JSON 파일 개수
    """

    frame_dir, json_dir = Path(frame_dir), Path(json_dir)

    # JSON 결과 폴더 초기화
    if json_dir.exists():                         # 기존 폴더가 있으면 삭제
        shutil.rmtree(json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)   # 새 폴더 생성

    # Detector (사람 검출기)와 Pose Estimator 초기화
    detector = init_detector(det_cfg, det_ckpt, device=device)   # 객체 탐지 모델 로드
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)            # mmpose 호환 파이프라인 적용

    pose_estimator = init_pose_estimator(                        # 포즈 추정 모델 초기화
        pose_cfg, pose_ckpt, device=device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))  # heatmap 비활성화
    )

    # 프레임 목록 수집
    frames = sorted(frame_dir.glob("*.jpg"))      # jpg 프레임 전체 정렬
    saved = 0                                    # 저장된 JSON 수 카운트

    # Batch 단위로 프레임 처리
    for start in tqdm(range(0, len(frames), batch_size), desc="Sapiens", unit="batch"):
        batch_files = frames[start:start + batch_size]           # 배치 단위 파일 목록
        batch_imgs_bgr = [cv2.imread(str(f)) for f in batch_files]  # BGR 이미지 읽기
        batch_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in batch_imgs_bgr if img is not None]  # RGB 변환

        if not batch_imgs:
            continue                                             # 비어 있으면 skip

        try:
            # 사람 감지 (Batch)
            dets = inference_detector(detector, batch_imgs)     # 사람 박스 예측

            # 각 프레임별 포즈 추정
            for idx_in_batch, (fpath, img_rgb, det) in enumerate(zip(batch_files, batch_imgs, dets)):
                idx_frame = start + idx_in_batch                 # 전체 프레임 인덱스 계산
                pred = det.pred_instances.cpu().numpy()          # 예측 결과 numpy 변환

                # 사람(label==0)만 추출 + confidence 0.2 이상 필터링
                keep = (pred.labels == 0) & (pred.scores > 0.7)
                bbs = np.concatenate((pred.bboxes, pred.scores[:, None]), axis=1)[keep]

                if len(bbs) == 0:
                    continue                                     # 사람 없음 → skip

                bbs = bbs[nms(bbs, 0.5), :4]                     # NMS 수행 (IoU 0.5)

                # 포즈 추정 (각 사람 bounding box별 keypoints 예측)
                pose_results = inference_topdown(pose_estimator, img_rgb, bbs)  # keypoints 추정
                data_sample = merge_data_samples(pose_results)   # 여러 사람 결과 통합
                inst = data_sample.get("pred_instances", None)
                if inst is None:
                    continue

                inst_list = split_instances(inst)                # 각 사람 instance 분리

                # ------------------------------------------------
                # 8️⃣ JSON 파일로 저장
                # ------------------------------------------------
                payload = dict(
                    frame_index=idx_frame,                       # 프레임 인덱스
                    meta_info=pose_estimator.dataset_meta,       # skeleton 구조 메타정보
                    instance_info=inst_list                      # 사람별 keypoints 정보
                )

                json_path = json_dir / f"{idx_frame:06d}.json"   # ex) 000123.json
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(to_py(payload), f, ensure_ascii=False, indent=2)
                saved += 1                                       # 저장 카운트 증가

        except Exception as e:
            print(f"[ERROR] batch {start} → {e}")                # 오류 시 배치 단위 경고

    # --------------------------------------------------------
    # 9️⃣ 총 저장된 JSON 개수 반환
    # --------------------------------------------------------
    return saved