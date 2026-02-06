import sys
import json
import shutil
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# --- MMPose 라이브러리 ---
try:
    from mmpose.apis import init_model
    from mmpose.utils import register_all_modules
    from mmpose.structures import PoseDataSample, merge_data_samples
    from mmengine.dataset import Compose
except ImportError:
    print("❌ MMPose 라이브러리가 필요합니다.")
    sys.exit(1)

try:
    from functions.extract_bbox_and_id import extract_bbox_and_id
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from functions.extract_bbox_and_id import extract_bbox_and_id

def to_py(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, dict): return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_py(v) for v in obj]
    return obj

# ============================================================
# 1️⃣ Dataset: Gray Padding (Letterbox) 적용 + Batch 준비
# ============================================================
class SapiensBatchDataset(Dataset):
    def __init__(self, tasks, frame_dir, input_size=(1024, 768)):
        self.frame_dir = Path(frame_dir)
        self.items = []
        self.input_size = input_size # (W, H)
        
        # SAM JSON에서 유효한 BBox가 있는 객체만 리스트업
        for sam_file, file_name, objects in tasks:
            f_idx = int(sam_file.stem) if sam_file.stem.isdigit() else 0
            for obj in objects:
                if obj.get('bbox'):
                    self.items.append({
                        'stem': sam_file.stem,
                        'file_name': file_name,
                        'frame_idx': f_idx, 
                        'obj_id': obj['id'], 
                        'bbox': obj['bbox'] # [x1, y1, x2, y2]
                    })

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = cv2.imread(str(self.frame_dir / item['file_name']))
        if img is None: return None
        
        # --- 1. Crop (SAM BBox + 1.2배 확장) ---
        x1, y1, x2, y2 = item['bbox']
        img_h, img_w = img.shape[:2]
        
        bw, bh = x2 - x1, y2 - y1
        cx, cy = x1 + bw / 2, y1 + bh / 2
        nw, nh = bw * 1.2, bh * 1.2 # 1.2배 확장
        
        # 이미지 경계 넘지 않도록 Clipping
        nx1, ny1 = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
        nx2, ny2 = min(img_w, int(cx + nw / 2)), min(img_h, int(cy + nh / 2))
        
        crop = img[ny1:ny2, nx1:nx2].copy()
        
        # --- 2. Gray Padding (Letterbox Resize) ---
        # 
        target_w, target_h = self.input_size
        h, w = crop.shape[:2]
        
        # 비율 유지 스케일 계산
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(crop, (new_w, new_h))
        
        # 회색(128) 캔버스 생성
        canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
        
        # 중앙 정렬
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        # --- 3. Normalize & ToTensor ---
        # MMPose 표준 Mean/Std 적용
        input_img = canvas.astype(np.float32) # 데이터를 소수점 연산이 가능한 float32 타입으로 변환합니다.
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32) # ImageNet 데이터셋의 RGB 채널별 평균값을 설정합니다.
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32) # ImageNet 데이터셋의 RGB 채널별 표준편차 값을 설정합니다.
        input_img = (input_img - mean) / std # (입력값 - 평균) / 표준편차 공식을 통해 데이터를 정규화(Normalization)합니다.
        input_img = input_img.transpose(2, 0, 1) # 이미지 배열 순서를 [높이, 너비, 채널]에서 [채널, 높이, 너비]로 변경합니다.
        
        # Meta 정보 (나중에 좌표 복원용)
        meta = {
            'stem': item['stem'],
            'file_name': item['file_name'],
            'frame_idx': item['frame_idx'],
            'obj_id': item['obj_id'],
            'crop_bbox': [nx1, ny1, nx2, ny2], # 원본 이미지에서의 Crop 위치
            'padding': [pad_x, pad_y],         # 추가된 패딩 양
            'scale_factor': scale,             # 리사이즈 비율
            'input_size': self.input_size      # 모델 입력 크기
        }
        
        return torch.from_numpy(input_img), meta

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]

# ============================================================
# 2️⃣ Inference: Full Model API 사용 + Batch 처리
# ============================================================
def run_sapiens_batch_inference(frame_dir, sam_dir, output_dir, config_path, ckpt_path, batch_size=8):
    frame_dir, sam_dir, output_dir = Path(frame_dir), Path(sam_dir), Path(output_dir)
    if output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    register_all_modules()
    
    # 1. SAM JSON 스캔
    sam_files = sorted(list(sam_dir.glob("*.json")))
    tasks = []
    print("📂 SAM JSON 스캔 중...")
    for sam_file in tqdm(sam_files):
        file_name, objects = extract_bbox_and_id(str(sam_file))
        if (frame_dir / file_name).exists():
            tasks.append((sam_file, file_name, objects))

    # 2. 모델 로드
    print("🚀 Sapiens 모델 로드 (Batch Mode)...")
    model = init_model(str(config_path), str(ckpt_path), device='cuda:0')
    model.eval()

    # 3. DataLoader 준비
    dataset = SapiensBatchDataset(tasks, frame_dir, input_size=(1024, 768))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    
    print(f"⚡ Batch Inference 시작 (Total Objects: {len(dataset)})")
    
    # 4. Inference Loop
    for batch in tqdm(loader, desc="Processing"):
        if batch is None: continue
        inputs, metas = batch
        inputs = inputs.to('cuda', non_blocking=True) # (B, 3, 768, 1024)

        # AMP 적용 (메모리 절약) Gradient 저장 하지 말고 BF16 계산법 사용
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            feats = model.extract_feat(inputs)
            
            # Head 예측
            batch_data_samples = [
                PoseDataSample(metainfo=dict(input_size=m['input_size'])) 
                for m in metas
            ]
            preds = model.head.predict(feats, batch_data_samples)

        # 5. 좌표 복원 및 저장
        for i, pred_sample in enumerate(preds):
            meta = metas[i]
            
            # 🌟 [수정] InstanceData vs PoseDataSample 호환 처리
            if hasattr(pred_sample, 'pred_instances'):
                # PoseDataSample 객체인 경우
                instances = pred_sample.pred_instances
            else:
                # InstanceData 객체 자체인 경우 (현재 에러 상황)
                instances = pred_sample
            
            # Keypoints 추출
            kpts_crop = instances.keypoints
            scores = instances.keypoint_scores
            
            # 차원 확인: (1, K, 2) 형태라면 배치 차원 제거
            if kpts_crop.ndim == 3:
                kpts_crop = kpts_crop[0]
                scores = scores[0]
            
            # --- 이하 좌표 복원 로직 동일 ---
            pad_x, pad_y = meta['padding']
            scale = meta['scale_factor']
            off_x, off_y = meta['crop_bbox'][:2]
            
            final_kpts = []
            # Tensor일 경우 CPU로 이동
            if isinstance(kpts_crop, torch.Tensor): kpts_crop = kpts_crop.cpu().numpy()
            if isinstance(scores, torch.Tensor): scores = scores.cpu().numpy()

            for (cx, cy), score in zip(kpts_crop, scores):
                # 1. 패딩 제거
                x_nopad = cx - pad_x
                y_nopad = cy - pad_y
                # 2. 스케일 복원
                fx = (x_nopad / scale) + off_x
                fy = (y_nopad / scale) + off_y
                final_kpts.append([float(fx), float(fy), float(score)])
            
            crop_bbox = [float(v) for v in meta['crop_bbox']]
            instance_item = {
                "instance_id": int(meta['obj_id']),
                "keypoints": final_kpts,
                "keypoint_scores": scores.tolist(),
                "bbox": crop_bbox
            }
            
            save_path = output_dir / f"{meta['stem']}.json"
            
            if save_path.exists():
                with open(save_path, "r", encoding="utf-8") as f:
                    data_j = json.load(f)
                data_j['instance_info'].append(instance_item)
            else:
                data_j = {
                    "frame_index": meta['frame_idx'],
                    "file_name": meta['file_name'],
                    "meta_info": to_py(model.dataset_meta),
                    "instance_info": [instance_item]
                }
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data_j, f, ensure_ascii=False, indent=2)
                
    return len(list(output_dir.glob("*.json")))

# ============================================================
# Main 실행부
# ============================================================
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from functions.generate_skeleton_video import generate_skeleton_video

if __name__ == "__main__":
    DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
    
    df = pd.read_csv(DATA_DIR / "metadata.csv")
    for target in range(705, 710):
        COMMON_PATH = df['common_path'][target]

        FRAME_DIR = DATA_DIR / "1_FRAME" / COMMON_PATH
        SAM_DIR = DATA_DIR / "8_SAM" / COMMON_PATH
        # v5.0: Full Model + Batch + SAM BBox + Gray Padding
        OUTPUT_DIR = DATA_DIR / "test" / COMMON_PATH / "v5.0_17kpt_full"

        # COCO 133점 기반    
        # CONFIG = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling/configs/sapiens/sapiens_0.3b-210e_coco_wholebody-1024x768.py"
        # CKPT = DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620.pth"

        # COCO 17점 기반
        CONFIG = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling/configs/sapiens/sapiens_0.3b-210e_coco-1024x768.py"
        CKPT = DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_best_coco_AP_796.pth"

        print(f"\noutput_dir: {OUTPUT_DIR}\n")
        # batch_size를 32~64 정도로 높여보세요 (VRAM 허용 시)
        # count = run_sapiens_batch_inference(FRAME_DIR, SAM_DIR, OUTPUT_DIR, CONFIG, CKPT, batch_size=20)
        # print(f"✅ 완료: {count}개 JSON 생성")

        # 결과 검증 영상 생성
        generate_skeleton_video(
            frame_dir=FRAME_DIR,
            kpt_dir=OUTPUT_DIR,
            output_path=str(f"{OUTPUT_DIR}_007_conf.mp4"),
            conf_threshold=0
        )