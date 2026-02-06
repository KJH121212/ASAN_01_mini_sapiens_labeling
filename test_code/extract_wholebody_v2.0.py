import sys
import os
import json
import shutil
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

# --- 1. 환경 설정 및 라이브러리 로드 ---
# 현재 파일 위치 기준 프로젝트 루트 설정 (모듈 import 문제 해결)
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent 
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 사용자 정의 함수 (BBox 추출용)
try:
    from functions.extract_bbox_and_id import extract_bbox_and_id
except ImportError:
    print("❌ 'functions' 모듈을 찾을 수 없습니다. 경로를 확인해주세요.")
    sys.exit(1)

# MMPose 및 관련 라이브러리 로드
try:
    from mmpose.apis import init_model, inference_topdown
    from mmengine import Config
    from mmpose.utils import register_all_modules
    # 최종 결과 포맷을 맞추기 위한 함수들
    from mmpose.structures import merge_data_samples, split_instances
except ImportError:
    print("❌ MMPose 라이브러리가 설치되지 않았습니다.")
    sys.exit(1)

# --- 2. 경로 설정 ---
data_path = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
base_path = project_root 

# 메타데이터 로드
df_path = data_path / "metadata.csv"
if not df_path.exists():
    print(f"❌ 메타데이터 파일 없음: {df_path}")
    sys.exit(1)
    
df = pd.read_csv(df_path)
common_path = df['common_path'][1] # 사용자가 지정한 인덱스 [1] 사용

checkpoint_path = data_path / "checkpoints/sapiens/pose/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620.pth"
frame_path = data_path / "1_FRAME" / common_path
sam_path = data_path / "8_SAM" / common_path
output_json_dir = base_path / "test" / "keypoints_result_v2.0" # 결과 저장 경로

# --- 3. Helper: Numpy -> JSON 변환 ---
def to_py(obj):
    """numpy 객체를 JSON 직렬화 가능한 Python 객체로 변환"""
    import numpy as _np
    if isinstance(obj, _np.ndarray): 
        return obj.tolist()
    if isinstance(obj, (_np.floating,)): 
        return float(obj)
    if isinstance(obj, (_np.integer,)):  
        return int(obj)
    if isinstance(obj, dict):  
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): 
        return [to_py(v) for v in obj]
    return obj

# --- 4. Helper: Padding & Crop 함수 (핵심) ---
def get_padded_crop(image, bbox, padding_ratio=0.2):
    """
    BBox 주변에 패딩을 추가하여 이미지를 크롭합니다.
    Args:
        image: 원본 이미지 (H, W, C)
        bbox: [x1, y1, x2, y2]
        padding_ratio: 패딩 비율 (기본 0.2 = 20%)
    Returns:
        crop_img: 잘라낸 이미지
        padded_bbox: [new_x1, new_y1, new_x2, new_y2] (좌표 원복용 오프셋)
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    # BBox 너비/높이 계산
    box_w = x2 - x1
    box_h = y2 - y1
    
    # 중심점 계산
    cx = x1 + box_w / 2
    cy = y1 + box_h / 2
    
    # 패딩 적용된 새 너비/높이 (20% 확장)
    new_w = box_w * (1 + padding_ratio)
    new_h = box_h * (1 + padding_ratio)
    
    # 새 좌표 계산 (정수형 변환)
    new_x1 = int(cx - new_w / 2)
    new_y1 = int(cy - new_h / 2)
    new_x2 = int(cx + new_w / 2)
    new_y2 = int(cy + new_h / 2)
    
    # 이미지 범위를 벗어나지 않도록 Clip 처리 (매우 중요)
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(w, new_x2)
    new_y2 = min(h, new_y2)
    
    # 유효성 검사: 크롭 영역이 없으면 None 반환
    if new_x2 <= new_x1 or new_y2 <= new_y1:
        return None, None

    # 이미지 크롭 (NumPy 슬라이싱)
    crop_img = image[new_y1:new_y2, new_x1:new_x2]
    
    # 잘린 영역의 좌표 반환 (나중에 원복할 때 필요)
    return crop_img, [new_x1, new_y1, new_x2, new_y2]

# --- 5. Helper: Sapiens Config 생성 (하드코딩 방식) ---
def get_sapiens_config(ckpt_path):
    """
    Config 파일을 읽지 않고 Python 코드로 직접 설정을 생성합니다.
    (경로 에러 및 아키텍처 미등록 에러를 원천 차단)
    """
    image_size = [768, 1024] # Width, Height
    
    # Pipeline 정의 (inference_topdown 함수가 필요로 함)
    test_pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=image_size, use_udp=True),
        dict(type='PackPoseInputs')
    ]

    model_cfg = dict(
        type='TopdownPoseEstimator',
        data_preprocessor=dict(
            type='PoseDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True
        ),
        backbone=dict(
            type='mmpretrain.VisionTransformer',
            # Sapiens 0.3b (ViT-Large) 아키텍처 사양 직접 주입
            arch=dict(
                embed_dims=1024, num_layers=24, num_heads=16, feedforward_channels=4096
            ),
            img_size=(image_size[1], image_size[0]),
            patch_size=16,
            qkv_bias=True,
            final_norm=True,
            out_type='featmap',
            with_cls_token=False, # Shape Mismatch 해결을 위해 False 설정
            patch_cfg=dict(padding=2),
            init_cfg=dict(type='Pretrained', checkpoint=str(ckpt_path)),
        ),
        head=dict(
            type='HeatmapHead',
            in_channels=1024,
            out_channels=133,
            # Weight Mismatch 해결을 위해 채널 수 명시 (768)
            deconv_out_channels=(768, 768),
            deconv_kernel_sizes=(4, 4),
            conv_out_channels=(768, 768),
            conv_kernel_sizes=(1, 1),
            loss=dict(type='KeypointMSELoss', use_target_weight=True),
            decoder=dict(
                type='UDPHeatmap',
                input_size=(image_size[0], image_size[1]),
                heatmap_size=(int(image_size[0]/4), int(image_size[1]/4)),
                sigma=6
            )
        ),
        test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=False)
    )
    
    # Dataloader에 Pipeline 주입 (ConfigDict Error 해결)
    dummy_dataloader = dict(
        dataset=dict(
            type='CocoWholeBodyDataset',
            pipeline=test_pipeline 
        )
    )
    gv
    return Config(dict(
        model=model_cfg, 
        test_dataloader=dummy_dataloader,
        default_scope='mmpose'
    ))

# --- 6. Main: Keypoints 추출 함수 ---
def extract_keypoints(frame_dir, sam_dir, output_dir, pose_ckpt, device='cuda:0'):
    # 출력 폴더 초기화 (기존 폴더 삭제 후 재생성)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 모델 초기화
    register_all_modules()
    print(f"🚀 Sapiens 모델 로드 중... ({device})")
    
    try:
        cfg = get_sapiens_config(pose_ckpt)
        pose_estimator = init_model(cfg, str(pose_ckpt), device=device)
        print("✅ 모델 로드 성공!")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return 0

    # 2. 파일 목록 준비 (SAM JSON 파일 기준)
    sam_dir = Path(sam_dir)
    frame_dir = Path(frame_dir)
    sam_files = sorted(list(sam_dir.glob("*.json")))
    print(f"📂 총 {len(sam_files)}개의 프레임 처리 예정")

    saved_count = 0

    # 3. 프레임별 반복 처리
    for sam_file in tqdm(sam_files, desc="Processing"):
        try:
            # SAM JSON에서 파일명과 객체 정보 추출
            file_name, objects = extract_bbox_and_id(str(sam_file))
            
            # 이미지 파일 확인 및 로드
            img_path = frame_dir / file_name
            if not img_path.exists(): continue
            
            img = cv2.imread(str(img_path))
            if img is None: continue

            # 해당 프레임의 모든 사람 결과(PoseDataSample)를 모을 리스트
            frame_pose_results = []

            # 4. 객체별 Loop (Detection 생략, SAM BBox 사용)
            for obj in objects:
                bbox = obj['bbox'] # [x1, y1, x2, y2]
                obj_id = obj['id']
                if not bbox: continue

                # [Step A] Padding & Crop
                # 원본 이미지에서 BBox 영역을 잘라냅니다.
                crop_img, padded_bbox = get_padded_crop(img, bbox, padding_ratio=0.2)
                if crop_img is None: continue

                # [Step B] Inference
                # 잘린 이미지(crop_img)를 모델에 넣습니다.
                # 이때 BBox는 이미지 전체 크기([0, 0, w, h])로 설정합니다.
                h_crop, w_crop = crop_img.shape[:2]
                input_bbox = np.array([0, 0, w_crop, h_crop])
                
                # Sapiens 추론 실행
                pose_results = inference_topdown(pose_estimator, crop_img, bboxes=input_bbox[None])
                
                # [Step C] 좌표 원복 (Remap)
                # 추론 결과는 잘린 이미지 기준 좌표이므로, 
                # 잘라낸 시작점(px1, py1)을 더해서 원본 좌표로 변환합니다.
                for res in pose_results:
                    # Keypoints 원복: [x, y] += [padding_x1, padding_y1]
                    res.pred_instances.keypoints[0] += [padded_bbox[0], padded_bbox[1]]
                    
                    # BBox 원복: Sapiens가 예측한 BBox도 원본 좌표계로 이동
                    res.pred_instances.bboxes[0] += [padded_bbox[0], padded_bbox[1], padded_bbox[0], padded_bbox[1]]

                    # ID 정보를 여기에 직접 넣을 수는 없지만(MMPose 구조상),
                    # 아래에서 split_instances 후 instance_info를 만들 때 SAM ID를 매핑해줄 수 있습니다.
                    
                    frame_pose_results.append(res)

            # 5. 결과 통합 및 저장 (MMPose 표준 포맷 준수)
            if frame_pose_results:
                # 여러 사람의 결과(DataSamples)를 하나로 병합
                data_sample = merge_data_samples(frame_pose_results)
                
                # 인스턴스 정보 추출
                inst = data_sample.get("pred_instances", None)
                if inst is not None:
                    # split_instances를 사용해 표준 dict 리스트로 변환
                    inst_list = split_instances(inst)
                    
                    # [중요] SAM 객체 순서와 inst_list 순서가 동일하다고 가정하고 ID 매핑
                    # (실제로는 위 loop 순서대로 append 했으므로 순서가 유지됩니다)
                    for i, item in enumerate(inst_list):
                        if i < len(objects):
                            item['instance_id'] = objects[i]['id'] # SAM ID 추가
                    
                    # 프레임 번호 추출 (예: 000123.jpg -> 123)
                    frame_idx = int(Path(file_name).stem) if Path(file_name).stem.isdigit() else 0
                    
                    # 최종 저장 데이터 구성
                    payload = dict(
                        frame_index=frame_idx,
                        file_name=file_name, # 원본 파일명도 저장하면 좋음
                        meta_info=pose_estimator.dataset_meta, # Skeleton 메타 정보
                        instance_info=inst_list                # Keypoints 정보
                    )
                    
                    # JSON 파일 저장
                    save_path = output_dir / f"{Path(file_name).stem}.json"
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(to_py(payload), f, ensure_ascii=False, indent=2)
                    
                    saved_count += 1

        except Exception as e:
            print(f"[Error] {sam_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return saved_count

if __name__ == "__main__":
    count = extract_keypoints(
        frame_dir=frame_path,
        sam_dir=sam_path,
        output_dir=output_json_dir,
        pose_ckpt=checkpoint_path
    )
    print(f"\n✅ 완료! 총 {count}개의 JSON 파일 생성됨.")
    print(f"📁 저장 경로: {output_json_dir}")