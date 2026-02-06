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

# --- 라이브러리 임포트 (기존과 동일) ---
try:
    from mmpose.apis import init_model
    from mmpose.utils import register_all_modules
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
# 1️⃣ Dataset: Crop 좌표(1.2배 + Clipping)를 meta에 저장
# ============================================================
class SapiensLiteDataset(Dataset):
    def __init__(self, tasks, frame_dir):
        self.frame_dir = Path(frame_dir)
        self.items = []
        self.input_res = (1024, 768) 
        for sam_file, file_name, objects in tasks:
            f_idx = int(sam_file.stem) if sam_file.stem.isdigit() else 0
            for obj in objects:
                if obj.get('bbox'):
                    self.items.append({
                        'stem': sam_file.stem, 'file_name': file_name,
                        'frame_idx': f_idx, 'obj_id': obj['id'], 'bbox': obj['bbox']
                    })

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = cv2.imread(str(self.frame_dir / item['file_name']))
        if img is None: return None
        
        # 1. SAM 원본 좌표
        x1, y1, x2, y2 = item['bbox']
        h, w = img.shape[:2]
        
        # 2. 1.2배 확장 계산
        bw, bh = x2 - x1, y2 - y1
        cx, cy = x1 + bw / 2, y1 + bh / 2
        nw, nh = bw * 1.2, bh * 1.2
        
        # 3. Clipping (이미지 밖으로 나가지 않게 자름) -> ⭐ 최종 저장할 좌표
        nx1, ny1 = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
        nx2, ny2 = min(w, int(cx + nw / 2)), min(h, int(cy + nh / 2))
        
        # Crop & Resize
        crop = img[ny1:ny2, nx1:nx2].copy()
        input_img = cv2.resize(crop, (self.input_res[1], self.input_res[0]))
        input_img = input_img.transpose(2, 0, 1).astype(np.float32)
        
        mean = np.array([123.675, 116.28, 103.53]).reshape(3, 1, 1).astype(np.float32)
        std = np.array([58.395, 57.12, 57.375]).reshape(3, 1, 1).astype(np.float32)
        input_img = (input_img - mean) / std
        
        return torch.from_numpy(input_img), {
            'stem': item['stem'], 
            'file_name': item['file_name'],
            'frame_idx': item['frame_idx'], 
            'obj_id': item['obj_id'],
            'offset': [nx1, ny1], 
            'scale': [crop.shape[1] / self.input_res[1], crop.shape[0] / self.input_res[0]],
            # ⭐ [핵심] 1.2배 확장되고 Clipping 된 실제 좌표를 저장
            'crop_bbox': [nx1, ny1, nx2, ny2] 
        }

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]

# ============================================================
# 2️⃣ Inference: meta의 'crop_bbox'를 그대로 사용
# ============================================================
def run_sapiens_lite_inference(frame_dir, sam_dir, output_dir, config_path, ckpt_path, batch_size=25):
    frame_dir, sam_dir, output_dir = Path(frame_dir), Path(sam_dir), Path(output_dir)
    if output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    register_all_modules()
    
    sam_files = sorted(list(sam_dir.glob("*.json")))
    tasks = []
    print("SAM JSON 스캔")
    for sam_file in tqdm(sam_files):
        file_name, objects = extract_bbox_and_id(str(sam_file))
        if (frame_dir / file_name).exists():
            tasks.append((sam_file, file_name, objects))

    print("Sapiens 모델 로드")
    model = init_model(str(config_path), str(ckpt_path), device='cuda:0')
    model.eval()

    dataset = SapiensLiteDataset(tasks, frame_dir)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False, collate_fn=collate_fn, pin_memory=True)

    print(f"Sapiens-Lite 추론")
    for batch in tqdm(loader, desc="Processing"):
        if batch is None: continue
        inputs, metas = batch
        inputs = inputs.to('cuda', non_blocking=True)

        with torch.no_grad():
            features = model.backbone(inputs)
            heatmaps = model.head(features)
            if isinstance(heatmaps, (list, tuple)): heatmaps = heatmaps[-1]

            B, C, H, W = heatmaps.shape
            heatmaps_reshaped = heatmaps.view(B, C, -1)
            max_vals, max_idxs = torch.max(heatmaps_reshaped, dim=2)
            preds_x = (max_idxs % W).float() * 4 
            preds_y = (max_idxs // W).float() * 4

            for i in range(B):
                meta = metas[i]
                
                final_x = (preds_x[i].cpu().numpy() * meta['scale'][0]) + meta['offset'][0]
                final_y = (preds_y[i].cpu().numpy() * meta['scale'][1]) + meta['offset'][1]
                scores = max_vals[i].cpu().numpy()
                keypoints_full = np.stack([final_x, final_y, scores], axis=1).tolist()

                # ⭐ [핵심] Dataset에서 넘겨준 crop_bbox 꺼내기
                crop_bbox = meta['crop_bbox']
                
                # Tensor -> List 변환 (DataLoader 배치 처리 시 Tensor로 변환됨)
                if isinstance(crop_bbox, torch.Tensor):
                    crop_bbox = crop_bbox.tolist()
                
                # 내부 값이 Tensor일 경우 float로 변환
                crop_bbox = [float(val) for val in crop_bbox]

                instance_item = {
                    "instance_id": int(meta['obj_id']),
                    "keypoints": keypoints_full,
                    "keypoint_scores": scores.tolist(),
                    "bbox": crop_bbox  # <--- 실제 Crop 영역 저장
                }

                save_path = output_dir / f"{meta['stem']}.json"
                if save_path.exists():
                    with open(save_path, "r") as f: data_j = json.load(f)
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

        torch.cuda.empty_cache()

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
    for target in range(1,6):
        COMMON_PATH = df['common_path'][target]

        FRAME_DIR = DATA_DIR / "1_FRAME" / COMMON_PATH
        SAM_DIR = DATA_DIR / "8_SAM" / COMMON_PATH
        OUTPUT_DIR = DATA_DIR / "test" / COMMON_PATH / "v4.1_17kpt_lite_0.3b"

        # COCO 133점 기반 0.3b
        # CONFIG = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling/configs/sapiens/sapiens_0.3b-210e_coco_wholebody-1024x768.py"
        # CKPT = DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620.pth"
        
        # COCO 133점 기반 0.6b
        # CONFIG = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling/configs/sapiens/sapiens_0.6b-210e_coco_wholebody-1024x768.py"
        # CKPT = DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.6b_coco_wholebody_best_coco_wholebody_AP_695.pth"

        # COCO 17점 기반
        CONFIG = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling/configs/sapiens/sapiens_0.3b-210e_coco-1024x768.py"
        CKPT = DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_best_coco_AP_796.pth"

        print(f"\noutput_dir: {OUTPUT_DIR}\n")
        count = run_sapiens_lite_inference(FRAME_DIR, SAM_DIR, OUTPUT_DIR, CONFIG, CKPT, batch_size=30)
        print(f"✅ 완료: {count}개 JSON 생성")

        generate_skeleton_video(
            frame_dir=FRAME_DIR,
            kpt_dir=OUTPUT_DIR,
            output_path=str(f"{OUTPUT_DIR}.mp4"),
            conf_threshold=0
            )