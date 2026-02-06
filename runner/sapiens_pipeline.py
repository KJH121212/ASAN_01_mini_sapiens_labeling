#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
process_video_from_metadata.py

📌 기능 요약:
- metadata.csv에서 각 비디오의 모든 경로(video_path, frame_path, keypoints_path, mp4_path 등)를 직접 읽음
- (1) 프레임 추출 → (2) Sapiens → (3) Reextract → (4) Overlay 자동 수행
- CSV 내 상태 컬럼 (frames_done, sapiens_done, reextract_done, overlay_done) 자동 갱신
"""

import sys
from pathlib import Path
import pandas as pd
from mmpose.apis import init_model as init_pose_estimator

# ============================================================
# 1️⃣ 기본 설정
# ============================================================
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling")
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
CSV_PATH = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/metadata.csv")


sys.path.append(str(BASE_DIR))
# 잘못된 버전 ❌
# from functions import extract_frames, extract_keypoints, reextract_missing_keypoints, render_skeleton_video

# 올바른 버전 ✅
from functions.extract_frames import extract_frames
from functions.extract_keypoints import extract_keypoints
from functions.reextract_missing_keypoints import reextract_missing_keypoints
from functions.render_skeleton_video import render_skeleton_video

# ============================================================
# 2️⃣ 단일 비디오 처리 함수 (metadata 기반)
# ============================================================
def process_video_from_metadata(row: dict):
    """metadata.csv의 단일 행(row) 기준으로 비디오 처리"""
    common_name    = row["common_path"]
    video_path     = Path(row["video_path"])
    frame_dir      = DATA_DIR / "1_FRAME" / common_name
    keypoint_dir   = DATA_DIR / "2_KEYPOINTS" / common_name
    mp4_path       = DATA_DIR / "3_MP4" / f"{common_name}.mp4"

    frame_dir.mkdir(parents=True, exist_ok=True)
    keypoint_dir.mkdir(parents=True, exist_ok=True)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    # 실행 여부
    run_frames     = not bool(row.get("frames_done", False))
    run_sapiens    = not bool(row.get("sapiens_done", False))
    run_reextract  = not bool(row.get("reextract_done", False))
    run_overlay    = not bool(row.get("overlay_done", False))

    n_frames = n_json = final_json_count = 0

    # --------------------------------------------------------
    # ① 프레임 추출
    # --------------------------------------------------------
    if run_frames:
        print("[STEP 1] 프레임 추출 중...")
        n_frames = extract_frames(str(video_path), str(frame_dir))
        row["frames_done"] = True
    else:
        print("[STEP 1] 프레임 추출 건너뜀")

    # --------------------------------------------------------
    # ② Sapiens keypoints 추출
    # --------------------------------------------------------
    if run_sapiens:
        print("[STEP 2] Sapiens 추출 중...")
        n_json = extract_keypoints(
            str(frame_dir), str(keypoint_dir),
            det_cfg  = str(BASE_DIR / "configs/detector/rtmdet_m_640-8xb32_coco-person_no_nms.py"),
            det_ckpt = str(DATA_DIR / "checkpoints/sapiens/detector/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"),
            pose_cfg = str(BASE_DIR / "configs/sapiens/sapiens_0.3b-210e_coco-1024x768.py"),
            pose_ckpt= str(DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_best_coco_AP_796.pth"),
            device="cuda:0"
        )
        print(f"    → 완료: {n_json} JSON files")
        row["sapiens_done"] = True
    else:
        print("[STEP 2] Sapiens 추출 건너뜀")

    # --------------------------------------------------------
    # ③ 누락 프레임 보정
    # --------------------------------------------------------
    if run_reextract:
        print("[STEP 3] 누락 프레임 보정 중...")
        if n_frames == 0 and frame_dir.exists():
            n_frames = len(list(frame_dir.glob("*.jpg")))

        pose_estimator = init_pose_estimator(
            str(BASE_DIR / "configs/sapiens/sapiens_0.3b-210e_coco-1024x768.py"),
            str(DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_best_coco_AP_796.pth"),
            device="cuda:0",
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
        )

        final_json_count = reextract_missing_keypoints(
            file_name = video_path.name,
            frame_dir = str(frame_dir),
            json_dir  = str(keypoint_dir),
            n_extracted_frames = n_frames,
            pose_estimator = pose_estimator
        )
        row["reextract_done"] = True
    else:
        print("[STEP 3] 누락 프레임 보정 건너뜀")

    # --------------------------------------------------------
    # ④ Overlay 생성
    # --------------------------------------------------------
    if run_overlay:
        print("[STEP 4] Overlay 영상 생성 중...")
        render_skeleton_video(str(frame_dir), str(keypoint_dir), str(mp4_path), fps=30)
        row["overlay_done"] = True
    else:
        print("[STEP 4] Overlay 건너뜀")

    # 프레임/JSON 개수 갱신
    row["n_frames"] = len(list(frame_dir.glob("*.jpg")))
    row["n_json"]   = len(list(keypoint_dir.glob("*.json")))

    return row


# ============================================================
# 3️⃣ 전체 metadata 순회 실행
# ============================================================
if __name__ == "__main__":
    if not CSV_PATH.exists():
        print(f"[ERROR] metadata.csv 없음 → {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] metadata.csv 내 총 비디오 개수: {len(df)}")

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        video_path = row_dict.get("video_path")

        try:
            updated = process_video_from_metadata(row_dict)
            for k, v in updated.items():
                df.at[idx, k] = v

            # ✅ 각 row 완료 후 즉시 저장
            df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
            print(f"[✅] {video_path} → 저장 완료 ({idx+1}/{len(df)})")
            print(f"[💾] metadata.csv 갱신됨 ({idx+1}/{len(df)})")

        except Exception as e:
            print(f"[❌ ERROR] {video_path} 처리 중 오류: {e}")
            continue

    print(f"\n[🏁 완료] 모든 비디오 처리 및 즉시 저장 완료 → {CSV_PATH}")
