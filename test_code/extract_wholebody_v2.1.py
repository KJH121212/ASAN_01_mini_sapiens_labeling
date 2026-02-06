import sys
import os
import json
import shutil
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# --- 라이브러리 임포트 (함수 내부에서 하지 않고 상단 배치) ---
try:
    from mmpose.apis import init_model, inference_topdown
    from mmengine import Config
    from mmpose.utils import register_all_modules
    from mmpose.structures import merge_data_samples, split_instances
except ImportError:
    print("❌ MMPose 라이브러리가 필요합니다.")
    sys.exit(1)

# 사용자 정의 함수 (BBox 추출용) - 경로 문제 발생 시 try-except
try:
    from functions.extract_bbox_and_id import extract_bbox_and_id
except ImportError:
    # 모듈 경로를 찾을 수 없는 경우를 대비해 현재 파일 상위 경로 추가
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent.parent 
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    try:
        from functions.extract_bbox_and_id import extract_bbox_and_id
    except ImportError:
        print("❌ 'functions' 모듈을 찾을 수 없습니다.")

# ============================================================
# 1️⃣ Helper Functions
# ============================================================
def to_py(obj):
    """numpy 객체를 JSON 직렬화 가능한 Python 객체로 변환"""
    import numpy as _np
    if isinstance(obj, _np.ndarray): return obj.tolist()
    if isinstance(obj, (_np.floating,)): return float(obj)
    if isinstance(obj, (_np.integer,)): return int(obj)
    if isinstance(obj, dict): return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_py(v) for v in obj]
    return obj

def get_padded_crop(image, bbox, padding_ratio=0.2):
    """BBox 주변에 패딩을 추가하여 이미지를 크롭"""
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

def get_sapiens_config(ckpt_path):
    """Sapiens Config 생성 (하드코딩)"""
    image_size = [768, 1024]
    
    test_pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=image_size, use_udp=True),
        dict(type='PackPoseInputs')
    ]

    model_cfg = dict(
        type='TopdownPoseEstimator',
        data_preprocessor=dict(type='PoseDataPreprocessor', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True),
        backbone=dict(
            type='mmpretrain.VisionTransformer',
            arch=dict(embed_dims=1024, num_layers=24, num_heads=16, feedforward_channels=4096),
            img_size=(image_size[1], image_size[0]), patch_size=16, qkv_bias=True, final_norm=True, out_type='featmap',
            with_cls_token=False, patch_cfg=dict(padding=2), init_cfg=dict(type='Pretrained', checkpoint=str(ckpt_path)),
        ),
        head=dict(
            type='HeatmapHead', in_channels=1024, out_channels=133,
            deconv_out_channels=(768, 768), deconv_kernel_sizes=(4, 4),
            conv_out_channels=(768, 768), conv_kernel_sizes=(1, 1),
            loss=dict(type='KeypointMSELoss', use_target_weight=True),
            decoder=dict(type='UDPHeatmap', input_size=(image_size[0], image_size[1]), heatmap_size=(int(image_size[0]/4), int(image_size[1]/4)), sigma=6)
        ),
        test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=False)
    )
    
    dummy_dataloader = dict(dataset=dict(type='CocoWholeBodyDataset', pipeline=test_pipeline))
    return Config(dict(model=model_cfg, test_dataloader=dummy_dataloader, default_scope='mmpose'))

# ============================================================
# 2️⃣ Main Function: Keypoints 추출
# ============================================================
def extract_keypoints(frame_dir: str, sam_dir: str, output_dir: str, 
                      pose_ckpt: str, device: str = 'cuda:0') -> int:
    """
    Sapiens 모델을 사용하여 Keypoints를 추출하고 JSON으로 저장하는 함수.
    
    Args:
        frame_dir (str): 원본 이미지가 있는 폴더 경로
        sam_dir (str): SAM 결과(BBox 포함) JSON 파일이 있는 폴더 경로
        output_dir (str): 결과를 저장할 폴더 경로
        pose_ckpt (str): Sapiens 모델 체크포인트(.pth) 파일 경로
        device (str): 실행 장치 (기본: 'cuda:0')
        
    Returns:
        int: 생성된 JSON 파일 개수
    """
    
    # 경로 객체 변환
    frame_dir = Path(frame_dir)
    sam_dir = Path(sam_dir)
    output_dir = Path(output_dir)
    
    # 출력 폴더 초기화
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

    # 2. 파일 목록 준비
    sam_files = sorted(list(sam_dir.glob("*.json")))
    print(f"📂 총 {len(sam_files)}개의 프레임 처리 예정")

    saved_count = 0

    # 3. 프레임별 반복 처리
    for sam_file in tqdm(sam_files, desc="Processing"):
        try:
            # SAM JSON 파싱 (파일명 및 BBox 추출)
            file_name, objects = extract_bbox_and_id(str(sam_file))
            
            # 이미지 로드
            img_path = frame_dir / file_name
            if not img_path.exists(): continue
            
            img = cv2.imread(str(img_path))
            if img is None: continue

            frame_pose_results = []

            # 4. 객체별 Loop
            for obj in objects:
                bbox = obj['bbox']
                if not bbox: continue

                # [Step A] Padding & Crop
                crop_img, padded_bbox = get_padded_crop(img, bbox, padding_ratio=0.2)
                if crop_img is None: continue

                # [Step B] Inference
                h_crop, w_crop = crop_img.shape[:2]
                input_bbox = np.array([0, 0, w_crop, h_crop])
                
                # Sapiens 추론
                pose_results = inference_topdown(pose_estimator, crop_img, bboxes=input_bbox[None])
                
                # [Step C] 좌표 원복 (Remap)
                for res in pose_results:
                    res.pred_instances.keypoints[0] += [padded_bbox[0], padded_bbox[1]]
                    res.pred_instances.bboxes[0] += [padded_bbox[0], padded_bbox[1], padded_bbox[0], padded_bbox[1]]
                    frame_pose_results.append(res)

            # 5. 결과 저장
            if frame_pose_results:
                data_sample = merge_data_samples(frame_pose_results)
                inst = data_sample.get("pred_instances", None)
                if inst is not None:
                    inst_list = split_instances(inst)
                    
                    # SAM ID 매핑
                    for i, item in enumerate(inst_list):
                        if i < len(objects):
                            item['instance_id'] = objects[i]['id']
                    
                    frame_idx = int(Path(file_name).stem) if Path(file_name).stem.isdigit() else 0
                    
                    payload = dict(
                        frame_index=frame_idx,
                        file_name=file_name,
                        meta_info=pose_estimator.dataset_meta,
                        instance_info=inst_list
                    )
                    
                    save_path = output_dir / f"{Path(file_name).stem}.json"
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(to_py(payload), f, ensure_ascii=False, indent=2)
                    
                    saved_count += 1

        except Exception as e:
            print(f"[Error] {sam_file.name}: {e}")
            continue

    return saved_count

# ============================================================
# 실행 예시 (직접 실행 시)
# ============================================================
if __name__ == "__main__":

    DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
    FRAME_DIR = DATA_DIR / "walking_data/FRAME/frontal__walking__1"
    SAM_DIR = DATA_DIR / "walking_data/sam/frontal__walking__1"
    OUTPUT_DIR = DATA_DIR / "walking_data/sapiens/frontal__walking__1"
    CKPT_PATH = DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620.pth"

    count = extract_keypoints(FRAME_DIR, SAM_DIR, OUTPUT_DIR, CKPT_PATH)
    print(f"✅ 완료: {count}개 파일 생성")

    FRAME_DIR = DATA_DIR / "walking_data/FRAME/lateral__walking__1"
    SAM_DIR = DATA_DIR / "walking_data/sam/lateral__walking__1"
    OUTPUT_DIR = DATA_DIR / "walking_data/sapiens/lateral__walking__1"
    CKPT_PATH = DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620.pth"

    count = extract_keypoints(FRAME_DIR, SAM_DIR, OUTPUT_DIR, CKPT_PATH)
    print(f"✅ 완료: {count}개 파일 생성")