import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from functions.run_sapiens_lite_inference import run_sapiens_lite_inference
from functions.generate_skeleton_video import generate_skeleton_video

import time

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
df = pd.read_csv(DATA_DIR / "metadata.csv")

for target in range(705,710):

    COMMON_PATH = df['common_path'][target]

    FRAME_DIR = DATA_DIR / "1_FRAME" / COMMON_PATH
    SAM_DIR = DATA_DIR / "8_SAM" / COMMON_PATH

    # COCO 133점 기반
    # SEG_OUTPUT_DIR = DATA_DIR / "9_KEYPOINTS_V2" / COMMON_PATH
    # VIDEO_OUTPUT_PATH = DATA_DIR / "10_VIDEO_V2" / f"{COMMON_PATH}.mp4"
    # CONFIG = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling/configs/sapiens/sapiens_0.3b-210e_coco_wholebody-1024x768.py"
    # CKPT = DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620.pth"

    # # COCO 17점 기반
    # SEG_OUTPUT_DIR = DATA_DIR / "test" / COMMON_PATH / "sapiens_lite_output"
    # VIDEO_OUTPUT_PATH = DATA_DIR / "test" / COMMON_PATH / "sapiens_lite_skeleton_video.mp4"
    # CONFIG = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling/configs/sapiens/sapiens_0.3b-210e_coco-1024x768.py"
    # CKPT = DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_best_coco_AP_796.pth"

    # total_start_time = time.time()

    # SAPIENS-Lite 추론
    # step1_start_time = time.time()
    # count = run_sapiens_lite_inference(
    #     FRAME_DIR, 
    #     SAM_DIR, 
    #     output_dir=SEG_OUTPUT_DIR, 
    #     config_path=CONFIG, 
    #     ckpt_path=CKPT, 
    #     batch_size=30
    # )
    
    # step1_end_time = time.time()
    # print(f"✅ 완료: {count}개 JSON 생성")

    # 비디오 생성
    # step2_start_time = time.time()
    SEG_OUTPUT_DIR = DATA_DIR / "2_KEYPOINTS" / COMMON_PATH
    # v5.0: Full Model + Batch + SAM BBox + Gray Padding
    OUTPUT_DIR = DATA_DIR / "test" / COMMON_PATH / "ORIGIN_17kpt.mp4"
    generate_skeleton_video(
        frame_dir=FRAME_DIR,
        kpt_dir=SEG_OUTPUT_DIR,
        output_path=OUTPUT_DIR,
        conf_threshold=0
    )
    # step2_end_time = time.time()

    # # 명시적 메모리 정리
    # import torch
    # import gc
    # torch.cuda.empty_cache()
    # gc.collect()

# =========================================================
# 📊 작업 정리 보고서 (개선된 가독성 버전)
# =========================================================

# # 소요 시간 계산
# step1_elapsed = step1_end_time - step1_start_time
# step2_elapsed = step2_end_time - step2_start_time
# total_elapsed = time.time() - total_start_time

# def format_time(seconds):
#     """초 단위 시간을 분/초 형태로 변환"""
#     return f"{int(seconds // 60)}m {seconds % 60:.2f}s"

# print("\n" + "="*100)
# print(f"📌 WORK SUMMARY REPORT | 대상: {COMMON_PATH}")
# print("="*100)

# # 헤더 출력
# print(f"{'작업 단계':<25} | {'결과물 수':<15} | {'소요 시간':<15}")
# print("-" * 100)

# # Step 1 출력
# print(f"{'1. Sapiens Lite 추론':<25} | {f'{count} JSONs':<15} | {format_time(step1_elapsed):<15}")
# print(f"   📂 경로: {SEG_OUTPUT_DIR}")
# print("-" * 100)

# # Step 2 출력
# print(f"{'2. 스켈레톤 비디오 생성':<25} | {'1 MP4':<15} | {format_time(step2_elapsed):<15}")
# print(f"   📂 경로: {VIDEO_OUTPUT_PATH}")
# print("-" * 100)

# # 전체 총계 출력
# print(f"{'⭐ 전체 총계':<25} | {'-':<15} | {format_time(total_elapsed):<15}")
# print("="*100)