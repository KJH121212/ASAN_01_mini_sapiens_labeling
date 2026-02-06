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

# --- 라이브러리 임포트 ---
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
# 1️⃣ Dataset: Gray Padding (Letterbox) 적용
# ============================================================
class SapiensLiteDataset(Dataset):
    def __init__(self, tasks, frame_dir):
        self.frame_dir = Path(frame_dir)
        self.items = []
        self.input_size = (1024, 768) 
        
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
        
        x1, y1, x2, y2 = item['bbox']
        img_h, img_w = img.shape[:2]
        
        # 1.2배 확장
        bw, bh = x2 - x1, y2 - y1
        cx, cy = x1 + bw / 2, y1 + bh / 2
        nw, nh = bw * 1.2, bh * 1.2
        
        # Clipping
        nx1, ny1 = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
        nx2, ny2 = min(img_w, int(cx + nw / 2)), min(img_h, int(cy + nh / 2))
        
        crop = img[ny1:ny2, nx1:nx2].copy()
        
        # Letterbox Padding
        target_w, target_h = self.input_size
        h, w = crop.shape[:2]
        
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(crop, (new_w, new_h))
        canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8) 
        
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        input_img = canvas.transpose(2, 0, 1).astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53]).reshape(3, 1, 1).astype(np.float32)
        std = np.array([58.395, 57.12, 57.375]).reshape(3, 1, 1).astype(np.float32)
        input_img = (input_img - mean) / std
        
        return torch.from_numpy(input_img), {
            'stem': item['stem'], 
            'file_name': item['file_name'],
            'frame_idx': item['frame_idx'], 
            'obj_id': item['obj_id'],
            'crop_offset': [nx1, ny1],
            'scale_factor': scale,
            'padding': [pad_x, pad_y],
            'crop_bbox': [nx1, ny1, nx2, ny2] 
        }

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]

# ============================================================
# 2️⃣ Helper: Sub-pixel Decoding 함수 (New!)
# ============================================================
def refine_keypoints_dark(heatmaps):
    """
    Standard "Dark" style refinement or Quarter-pixel shift
    히트맵의 Peak 주변 값을 비교하여 0.25 픽셀 단위로 미세 조정
    """
    B, K, H, W = heatmaps.shape
    
    # 1. Flatten해서 최대값 인덱스(Integer) 찾기
    heatmaps_flat = heatmaps.view(B, K, -1)
    max_vals, max_idxs = torch.max(heatmaps_flat, dim=2)
    
    # 정수 좌표 (Integer Coordinates)
    px_int = (max_idxs % W).long()
    py_int = (max_idxs // W).long()
    
    # 2. Boundary Check (가장자리에 있으면 보정 불가 -> 1칸 안쪽으로 clamp)
    px_clamped = px_int.clamp(1, W - 2)
    py_clamped = py_int.clamp(1, H - 2)
    
    # 3. 주변 픽셀 값 가져오기 (Gather)
    # Batch와 Keypoint 차원을 유지하기 위한 인덱싱
    b_idx = torch.arange(B)[:, None, None].to(heatmaps.device) # shape (B, 1, 1)
    k_idx = torch.arange(K)[None, :, None].to(heatmaps.device) # shape (1, K, 1)
    
    # 좌우상하 값 추출
    # heatmaps shape: [B, K, H, W]
    # indexing: [B, K, 1] 형태로 값을 뽑아냄 -> squeeze로 (B, K) 만듦
    val_r = heatmaps[torch.arange(B)[:, None], torch.arange(K)[None, :], py_clamped, px_clamped + 1]
    val_l = heatmaps[torch.arange(B)[:, None], torch.arange(K)[None, :], py_clamped, px_clamped - 1]
    val_d = heatmaps[torch.arange(B)[:, None], torch.arange(K)[None, :], py_clamped + 1, px_clamped]
    val_u = heatmaps[torch.arange(B)[:, None], torch.arange(K)[None, :], py_clamped - 1, px_clamped]

    # 4. Shift 계산 (Gradient 방향으로 이동)
    # 오른쪽이 왼쪽보다 크면 +0.25 (오른쪽으로 치우침)
    # 왼쪽이 오른쪽보다 크면 -0.25
    dx = torch.sign(val_r - val_l) * 0.25
    dy = torch.sign(val_d - val_u) * 0.25
    
    # 5. 최종 좌표: 정수 좌표 + 보정값
    # (Clamp된 좌표가 아니라 원래 찾았던 px_int, py_int에 더해야 함)
    pred_x = px_int.float() + dx
    pred_y = py_int.float() + dy
    
    return pred_x, pred_y, max_vals


# ============================================================
# 3️⃣ Inference Loop
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

    print(f"Sapiens-Lite (Gray Padding + Sub-pixel Decoding) 추론")
    
    for batch in tqdm(loader, desc="Processing"):
        if batch is None: continue
        inputs, metas = batch
        inputs = inputs.to('cuda', non_blocking=True)

        with torch.no_grad():
            features = model.backbone(inputs)
            heatmaps = model.head(features)
            if isinstance(heatmaps, (list, tuple)): heatmaps = heatmaps[-1]

            # 🌟 [핵심 변경] Sub-pixel 좌표 계산 함수 호출
            # 기존: argmax -> 정수
            # 변경: refine_keypoints_dark -> 실수 (소수점 포함)
            preds_x_hat, preds_y_hat, max_vals = refine_keypoints_dark(heatmaps)
            
            # Heatmap Stride (4배) 복원
            preds_x = preds_x_hat * 4
            preds_y = preds_y_hat * 4

            B = preds_x.shape[0]
            for i in range(B):
                meta = metas[i]
                
                # Tensor -> Numpy
                px = preds_x[i].cpu().numpy()
                py = preds_y[i].cpu().numpy()
                scores = max_vals[i].cpu().numpy()
                
                pad_x, pad_y = meta['padding']
                scale = meta['scale_factor']
                off_x, off_y = meta['crop_offset']
                
                # 1. 패딩 제거
                x_nopad = px - pad_x
                y_nopad = py - pad_y
                
                # 2. 스케일 복원 & 오프셋 추가 (실수 연산 유지)
                final_x = (x_nopad / scale) + off_x
                final_y = (y_nopad / scale) + off_y
                
                keypoints_full = np.stack([final_x, final_y, scores], axis=1).tolist()

                crop_bbox = meta['crop_bbox']
                if isinstance(crop_bbox, torch.Tensor): crop_bbox = crop_bbox.tolist()
                crop_bbox = [float(val) for val in crop_bbox]

                instance_item = {
                    "instance_id": int(meta['obj_id']),
                    "keypoints": keypoints_full,
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
    for target in range(0,5):
        COMMON_PATH = df['common_path'][target]

        FRAME_DIR = DATA_DIR / "1_FRAME" / COMMON_PATH
        SAM_DIR = DATA_DIR / "8_SAM" / COMMON_PATH
        OUTPUT_DIR = DATA_DIR / "test" / COMMON_PATH / "v4.4_17kpt_lite_0.3b"

        # COCO 133점 기반    
        # CONFIG = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling/configs/sapiens/sapiens_0.3b-210e_coco_wholebody-1024x768.py"
        # CKPT = DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620.pth"

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