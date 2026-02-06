import sys
import json
import torch
import numpy as np
import pandas as pd
import time
import copy
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# --- MMPose / MMEngine 라이브러리 임포트 ---
try:
    from mmpose.apis import init_model
    from mmpose.utils import register_all_modules
    from mmcv.transforms import Compose
    from mmengine.dataset import pseudo_collate
except ImportError:
    print("❌ MMPose 라이브러리가 설치되어 있지 않습니다. Full Installation 환경인지 확인하세요.")
    sys.exit(1)

# --- 사용자 정의 함수 경로 설정 ---
sys.path.append(str(Path(__file__).resolve().parent.parent))
from functions.generate_skeleton_video import generate_skeleton_video

# =========================================================
# 🛠️ 1. Custom Dataset (input_center/scale 추가 수정됨)
# =========================================================
class SapiensBatchDataset(Dataset):
    def __init__(self, sam_dir, frame_dir, pipeline_cfg, dataset_meta):
        self.frame_dir = Path(frame_dir)
        self.sam_files = sorted(list(Path(sam_dir).glob("*.json")))
        self.items = []
        self.dataset_meta = dataset_meta
        
        # [핵심 수정] 파이프라인 설정 깊은 복사
        pipeline_cfg = copy.deepcopy(pipeline_cfg)
        
        # ⭐ 'flip_indices', 'input_center', 'input_scale' 필수!
        # 이 키들이 있어야 좌표 복원(decoding) 과정에서 에러가 안 납니다.
        custom_keys = [
            'sam_stem', 'file_name', 'instance_id', 'raw_bbox', 
            'flip_indices', 'dataset_name',
            'input_center', 'input_scale', 'input_size' # <-- 추가됨
        ]
        
        found_pack = False
        for step in pipeline_cfg:
            if step['type'] == 'PackPoseInputs':
                # 기존 meta_keys 가져오기 (없으면 기본값 사용)
                default_meta_keys = ('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction')
                current_keys = list(step.get('meta_keys', default_meta_keys))
                
                # 커스텀 키 추가 (중복 방지)
                for k in custom_keys:
                    if k not in current_keys:
                        current_keys.append(k)
                
                step['meta_keys'] = tuple(current_keys)
                found_pack = True
                break
        
        if not found_pack:
            print("⚠️ Warning: 'PackPoseInputs' step not found. Metadata might be lost.")

        self.pipeline = Compose(pipeline_cfg)

        print(f"📂 데이터셋 로드 중... (SAM 파일 스캔)")
        for sam_file in tqdm(self.sam_files, desc="Scanning Metadata"):
            with open(sam_file, 'r') as f:
                data = json.load(f)
            
            file_name = data.get('file_name', sam_file.stem + ".jpg")
            img_path = self.frame_dir / file_name
            if not img_path.exists():
                img_path = self.frame_dir / (sam_file.stem + ".png")
                if not img_path.exists(): continue
            
            objects = data.get('objects', []) if 'objects' in data else data.get('instance_info', [])
            
            for obj in objects:
                bbox = obj.get('bbox')
                if not bbox: continue
                if isinstance(bbox[0], list): bbox = bbox[0]
                
                self.items.append({
                    'img_path': str(img_path),
                    'bbox': np.array(bbox, dtype=np.float32).reshape(1, 4),
                    'bbox_score': np.ones(1, dtype=np.float32),
                    'category_id': 1,
                    # 메타데이터
                    'sam_stem': sam_file.stem,
                    'file_name': file_name,
                    'instance_id': obj.get('id', obj.get('instance_id', -1)),
                    'raw_bbox': bbox
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        data_info = self.items[idx].copy()
        if self.dataset_meta:
            data_info.update(self.dataset_meta)
        return self.pipeline(data_info)

# =========================================================
# 🚀 2. 메인 배치 추론 함수
# =========================================================
def run_sapiens_full_batch_inference(
    frame_dir, sam_dir, output_dir, 
    pose_config, pose_ckpt, 
    batch_size=32, 
    device='cuda:0'
):
    frame_dir, sam_dir, output_dir = Path(frame_dir), Path(sam_dir), Path(output_dir)
    
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    register_all_modules()
    print(f"📦 Pose 모델 로드 중...\nConfig: {Path(pose_config).name}")
    model = init_model(pose_config, pose_ckpt, device=device)
    
    dataset_meta = model.dataset_meta
    if dataset_meta and 'flip_indices' not in dataset_meta:
        print("⚠️ Warning: 'flip_indices' not found in model meta. Augmentation might fail.")

    pipeline_cfg = model.cfg.test_dataloader.dataset.pipeline
    dataset = SapiensBatchDataset(sam_dir, frame_dir, pipeline_cfg, dataset_meta)
    
    if len(dataset) == 0:
        print("⚠️ 처리할 유효한 데이터가 없습니다.")
        return 0

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8, 
        collate_fn=pseudo_collate, 
        pin_memory=True
    )

    print(f"🔥 Full-Batch 추론 시작 (총 {len(dataset)} 객체 / Batch: {batch_size})")
    
    results_by_frame = {} 

    for batch in tqdm(dataloader, desc="Inference"):
        with torch.no_grad():
            batch_results = model.test_step(batch)

        for i, data_sample in enumerate(batch_results):
            metainfo = data_sample.metainfo
            sam_stem = metainfo.get('sam_stem', 'unknown')
            
            pred_instances = data_sample.pred_instances
            
            if isinstance(pred_instances.keypoints, torch.Tensor):
                keypoints = pred_instances.keypoints[0].cpu().numpy()
            else:
                keypoints = pred_instances.keypoints[0]

            if isinstance(pred_instances.keypoint_scores, torch.Tensor):
                keypoint_scores = pred_instances.keypoint_scores[0].cpu().numpy()
            else:
                keypoint_scores = pred_instances.keypoint_scores[0]
            
            kpts_list = []
            for (x, y), score in zip(keypoints, keypoint_scores):
                kpts_list.append([float(x), float(y), float(score)])

            instance_info = {
                "instance_id": metainfo.get('instance_id', -1),
                "bbox": metainfo.get('raw_bbox', []),
                "score": 1.0, 
                "keypoints": kpts_list,
                "keypoint_scores": keypoint_scores.tolist()
            }
            
            if sam_stem not in results_by_frame:
                results_by_frame[sam_stem] = {
                    "file_name": metainfo.get('file_name', f"{sam_stem}.jpg"),
                    "frame_index": int(sam_stem) if sam_stem.isdigit() else 0,
                    "instance_info": []
                }
            results_by_frame[sam_stem]["instance_info"].append(instance_info)

    print("💾 결과 JSON 저장 중...")
    
    model_meta = {
        "dataset_name": "coco_wholebody" if "133" in str(pose_config) else "coco",
        "num_keypoints": model.dataset_meta.get('num_keypoints', 133),
        "skeleton_links": model.dataset_meta.get('skeleton_links', []),
        "keypoint_name2id": model.dataset_meta.get('keypoint_name2id', {})
    }

    count = 0
    for stem, data in results_by_frame.items():
        final_json = {
            "file_name": data['file_name'],
            "frame_index": data['frame_index'],
            "meta_info": model_meta,
            "instance_info": data['instance_info']
        }
        
        save_path = output_dir / f"{stem}.json"
        with open(save_path, "w", encoding='utf-8') as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)
        count += 1

    return count

# =========================================================
# 🎬 실행부 (Main)
# =========================================================
if __name__ == "__main__":
    DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
    BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling")

    df = pd.read_csv(DATA_DIR / "metadata.csv")
    COMMON_PATH = df['common_path'][7]

    FRAME_DIR = DATA_DIR / "1_FRAME" / COMMON_PATH
    SAM_DIR = DATA_DIR / "8_SAM" / COMMON_PATH
    
    SEG_OUTPUT_DIR = DATA_DIR / "test" / COMMON_PATH / "sapiens_full_batch_output_133"
    VIDEO_OUTPUT_PATH = DATA_DIR / "test" / COMMON_PATH / "sapiens_full_batch_video_133.mp4"

    # POSE_CONFIG = str(BASE_DIR / "configs/sapiens/sapiens_0.3b-210e_coco-1024x768.py")
    # POSE_CKPT = str(DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_best_coco_AP_796.pth")
    POSE_CONFIG = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling/configs/sapiens/sapiens_0.3b-210e_coco_wholebody-1024x768.py"
    POSE_CKPT = str(DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620.pth")

    print(f"🚀 작업 시작: {COMMON_PATH}")
    total_start_time = time.time()

    # Step 1
    step1_start = time.time()
    n_json = run_sapiens_full_batch_inference(
        FRAME_DIR, SAM_DIR, SEG_OUTPUT_DIR, POSE_CONFIG, POSE_CKPT,
        batch_size=30, device="cuda:0"
    )
    step1_end = time.time()

    # Step 2
    step2_start = time.time()
    video_res = "건너뜀"
    if n_json > 0:
        generate_skeleton_video(FRAME_DIR, SEG_OUTPUT_DIR, VIDEO_OUTPUT_PATH, conf_threshold=0)
        video_res = "완료"
    step2_end = time.time()

    # Report
    step1_elapsed = step1_end - step1_start
    step2_elapsed = step2_end - step2_start
    total_elapsed = time.time() - total_start_time

    def format_time(s): return f"{int(s//60)}m {s%60:.2f}s"

    print("\n" + "="*100)
    print(f"📌 BATCH WORK SUMMARY | 대상: {COMMON_PATH}")
    print("="*100)
    print(f"{'작업 단계':<25} | {'결과물 수':<15} | {'소요 시간':<15}")
    print("-" * 100)
    print(f"{'1. Sapiens Full Batch':<25} | {f'{n_json} JSONs':<15} | {format_time(step1_elapsed):<15}")
    print(f"   📂 경로: {SEG_OUTPUT_DIR}")
    print("-" * 100)
    print(f"{'2. 스켈레톤 비디오 생성':<25} | {video_res:<15} | {format_time(step2_elapsed):<15}")
    print(f"   📂 경로: {VIDEO_OUTPUT_PATH}")
    print("-" * 100)
    print(f"{'⭐ 전체 총계':<25} | {'-':<15} | {format_time(total_elapsed):<15}")
    print("="*100)