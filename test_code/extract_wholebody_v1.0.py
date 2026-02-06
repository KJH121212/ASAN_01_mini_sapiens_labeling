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
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent 
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from functions.extract_bbox_and_id import extract_bbox_and_id
except ImportError:
    print("❌ 'functions' 모듈을 찾을 수 없습니다. 경로를 확인해주세요.")
    sys.exit(1)

try:
    from mmpose.apis import init_model, inference_topdown
    from mmengine import Config
    from mmpose.utils import register_all_modules
except ImportError:
    print("❌ MMPose 라이브러리가 설치되지 않았습니다.")
    sys.exit(1)

# --- 2. 경로 설정 ---
data_path = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
base_path = project_root 

df_path = data_path / "metadata.csv"
if not df_path.exists():
    print(f"❌ 메타데이터 파일 없음: {df_path}")
    sys.exit(1)
    
df = pd.read_csv(df_path)
common_path = df['common_path'][1]

checkpoint_path = data_path / "checkpoints/sapiens/pose/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620.pth"
frame_path = data_path / "1_FRAME" / common_path
sam_path = data_path / "8_SAM" / common_path
output_json_dir = base_path / "test" / "keypoints_result"

# --- 3. Helper Functions ---
def to_py(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer, int)): return int(obj)
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, dict): return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list): return [to_py(v) for v in obj]
    return obj

def get_padded_crop(image, bbox, padding_ratio=0.2):
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    box_w = x2 - x1
    box_h = y2 - y1
    cx = x1 + box_w / 2
    cy = y1 + box_h / 2
    
    new_w = box_w * (1 + padding_ratio)
    new_h = box_h * (1 + padding_ratio)
    
    new_x1 = int(cx - new_w / 2)
    new_y1 = int(cy - new_h / 2)
    new_x2 = int(cx + new_w / 2)
    new_y2 = int(cy + new_h / 2)
    
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(w, new_x2)
    new_y2 = min(h, new_y2)
    
    if new_x2 <= new_x1 or new_y2 <= new_y1:
        return None, None

    crop_img = image[new_y1:new_y2, new_x1:new_x2]
    return crop_img, [new_x1, new_y1, new_x2, new_y2]

# --- 4. Helper: Sapiens Config 생성 (Complete Fix) ---
def get_sapiens_config(ckpt_path):
    """
    Sapiens 0.3b Config 생성.
    1. with_cls_token=False (Shape Mismatch 해결)
    2. Head Channels=768 (Weight Mismatch 해결)
    3. Pipeline 추가 ('ConfigDict' pipeline 에러 해결)
    """
    image_size = [768, 1024]
    
    # 1. Pipeline 정의 (에러 방지용 필수 항목)
    test_pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=image_size, use_udp=True),
        dict(type='PackPoseInputs')
    ]

    # 2. 모델 구조 설정
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
            arch=dict(
                embed_dims=1024,
                num_layers=24,
                num_heads=16,
                feedforward_channels=4096
            ),
            img_size=(image_size[1], image_size[0]),
            patch_size=16,
            qkv_bias=True,
            final_norm=True,
            out_type='featmap',
            with_cls_token=False,  # [중요] Shape Mismatch 해결
            patch_cfg=dict(padding=2),
            init_cfg=dict(type='Pretrained', checkpoint=str(ckpt_path)),
        ),
        head=dict(
            type='HeatmapHead',
            in_channels=1024,
            out_channels=133,
            # [중요] Weight Mismatch 해결 (원본 Config 값 적용)
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
    
    # 3. Dataloader에 Pipeline 주입
    dummy_dataloader = dict(
        dataset=dict(
            type='CocoWholeBodyDataset',
            pipeline=test_pipeline # [중요] Pipeline 에러 해결
        )
    )
    
    return Config(dict(
        model=model_cfg, 
        test_dataloader=dummy_dataloader,
        default_scope='mmpose'
    ))

# --- 5. Main Execution ---
def extract_keypoints(frame_dir, sam_dir, output_dir, pose_ckpt, device='cuda:0'):
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
        print("✅ 모델 로드 성공! (무결성 확인 완료)")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return 0

    # 2. SAM 파일 처리
    sam_dir = Path(sam_dir)
    frame_dir = Path(frame_dir)
    sam_files = sorted(list(sam_dir.glob("*.json")))
    print(f"📂 총 {len(sam_files)}개의 프레임 처리 예정")

    saved_count = 0

    for sam_file in tqdm(sam_files, desc="Processing"):
        try:
            file_name, objects = extract_bbox_and_id(str(sam_file))
            
            img_path = frame_dir / file_name
            if not img_path.exists(): continue
            
            img = cv2.imread(str(img_path))
            if img is None: continue

            instance_list = []

            # 3. Pose Estimation Loop
            for obj in objects:
                bbox = obj['bbox']
                obj_id = obj['id']
                if not bbox: continue

                # A. Padding & Crop
                crop_img, padded_bbox = get_padded_crop(img, bbox, padding_ratio=0.2)
                if crop_img is None: continue

                # B. Inference
                h_crop, w_crop = crop_img.shape[:2]
                input_bbox = np.array([0, 0, w_crop, h_crop])
                
                pose_results = inference_topdown(pose_estimator, crop_img, bboxes=input_bbox[None])
                
                # C. Result Processing
                for res in pose_results:
                    kpts = res.pred_instances.keypoints[0]
                    kpts[:, 0] += padded_bbox[0]
                    kpts[:, 1] += padded_bbox[1]
                    
                    scores = res.pred_instances.keypoint_scores[0]

                    instance_info = {
                        "instance_id": obj_id,
                        "bbox": bbox,
                        "score": float(np.mean(scores)),
                        "keypoints": kpts,
                        "keypoint_scores": scores
                    }
                    instance_list.append(instance_info)

            # 4. JSON Save
            if instance_list:
                frame_idx = int(Path(file_name).stem) if Path(file_name).stem.isdigit() else 0
                
                meta_info = getattr(pose_estimator, 'dataset_meta', {})
                
                payload = dict(
                    frame_index=frame_idx,
                    file_name=file_name,
                    meta_info=meta_info,
                    instance_info=instance_list
                )
                
                save_path = output_dir / f"{Path(file_name).stem}.json"
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(to_py(payload), f, ensure_ascii=False, indent=2)
                
                saved_count += 1

        except Exception as e:
            print(f"[Error] {sam_file.name}: {e}")
            import traceback
            traceback.print_exc() # 에러 상세 출력
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