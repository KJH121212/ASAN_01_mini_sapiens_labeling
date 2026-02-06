import sys
import json
import shutil
import cv2
import numpy as np
import pandas as pd
import gc
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

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
# 0️⃣ [NEW] BBox 안정화 매니저 (핵심 로직)
# ============================================================
class DynamicBBoxManager:
    def __init__(self, expand_ratio=1.0, min_occupancy=0.7, smoothing=0.2, cooldown=30):
        """
        Args:
            expand_ratio (float): 확장 비율 (기본 1.2배)
            min_occupancy (float): 최소 점유율 (0.7 = 70%). 이보다 작아지면(공간낭비) 축소.
            smoothing (float): 변경 시 부드러운 이동 계수 (EMA)
            cooldown (int): 잦은 크기 변경 방지용 타이머
        """
        self.expand_ratio = expand_ratio
        # 점유율 70% 이하 == 여백 30% 이상 (1.0 - 0.7 = 0.3)
        self.margin_threshold = 1.0 - min_occupancy 
        self.alpha = smoothing
        
        # 상태 변수
        self.current_box = None 
        self.cooldown_max = cooldown
        self.cooldown_timer = 0

    def update(self, mask_bbox):
        # 1. 초기화 (첫 프레임)
        if self.current_box is None:
            self.current_box = self._make_target_box(mask_bbox, self.expand_ratio)
            return self.current_box

        # 쿨타임 감소
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1

        # 2. 상태 판단 (우선순위: 확장 > 축소 > 유지)
        
        # [Priority 1] 확장: 마스크가 박스를 뚫고 나감 (객체 잘림 방지) -> 즉시 실행
        if self._is_out_of_bound(mask_bbox, self.current_box):
            target_box = self._make_target_box(mask_bbox, self.expand_ratio)
            self.current_box = self._smooth_update(self.current_box, target_box)
            self.cooldown_timer = self.cooldown_max # 변경 후 휴식
            
        # [Priority 2] 축소: 점유율이 70% 이하로 떨어짐 (공간 낭비) -> 쿨타임 종료 후 실행
        elif self.cooldown_timer == 0 and \
             self._has_excessive_margin(mask_bbox, self.current_box, self.margin_threshold):
            
            target_box = self._make_target_box(mask_bbox, self.expand_ratio)
            self.current_box = self._smooth_update(self.current_box, target_box)
            self.cooldown_timer = self.cooldown_max
            
        # [Priority 3] 유지 (Dead Zone): Jitter 방지 구간
        else:
            pass # 아무것도 안 함 (현재 박스 고정)

        return self.current_box

    # --- 내부 로직 ---
    def _make_target_box(self, box, ratio):
        """중심 기준 확장"""
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        nw, nh = w * ratio, h * ratio
        return [cx - nw/2, cy - nh/2, cx + nw/2, cy + nh/2]

    def _is_out_of_bound(self, mask, view):
        """마스크가 뷰 박스를 벗어났는가?"""
        pad = 0.1 # 부동소수점 오차 방지
        return (mask[0] < view[0]-pad) or (mask[1] < view[1]-pad) or \
               (mask[2] > view[2]+pad) or (mask[3] > view[3]+pad)

    def _has_excessive_margin(self, mask, view, threshold):
        """빈 공간 비율이 임계치(30%)를 넘는가?"""
        view_w = view[2] - view[0]
        view_h = view[3] - view[1]
        mask_w = mask[2] - mask[0]
        mask_h = mask[3] - mask[1]
        
        if view_w <= 0 or view_h <= 0: return True
        
        # 가로 여백 or 세로 여백 중 하나라도 크면 True
        margin_w = 1.0 - (mask_w / view_w)
        margin_h = 1.0 - (mask_h / view_h)
        return (margin_w > threshold) or (margin_h > threshold)

    def _smooth_update(self, current, target):
        """EMA 스무딩"""
        return [c * (1 - self.alpha) + t * self.alpha for c, t in zip(current, target)]

def stabilize_bboxes(tasks):
    """전체 태스크를 순회하며 BBox를 안정화된 값으로 덮어씌움"""
    print("🌊 BBox 안정화 로직 적용 중 (Hysteresis)...")
    managers = {} # {obj_id: Manager}

    # tasks는 시간순 정렬되어 있다고 가정
    for sam_file, file_name, objects in tasks:
        for obj in objects:
            if not obj.get('bbox'): continue
            
            oid = obj['id']
            raw_bbox = obj['bbox']
            
            # ID별 매니저 생성 (1.2배 확장, 70% 점유율 유지, 스무딩 0.2)
            if oid not in managers:
                managers[oid] = DynamicBBoxManager(expand_ratio=1.1, min_occupancy=0.7, smoothing=0.2)
            
            # 안정화된 박스 계산
            stable_bbox = managers[oid].update(raw_bbox)
            
            # 🌟 원본 데이터 교체!
            obj['bbox'] = stable_bbox
            
    return tasks

# ============================================================
# 1️⃣ Dataset: Gray Padding (Letterbox) 적용 + Batch 준비
# ============================================================
class SapiensBatchDataset(Dataset):
    def __init__(self, tasks, frame_dir, input_size=(1024, 768)):
        self.frame_dir = Path(frame_dir)
        self.items = []
        self.input_size = input_size # (W, H)
        
        # 이미 stabilize_bboxes를 거쳐서 bbox가 보정된 tasks를 받음
        for sam_file, file_name, objects in tasks:
            f_idx = int(sam_file.stem) if sam_file.stem.isdigit() else 0
            for obj in objects:
                if obj.get('bbox'):
                    self.items.append({
                        'stem': sam_file.stem,
                        'file_name': file_name,
                        'frame_idx': f_idx, 
                        'obj_id': obj['id'], 
                        'bbox': obj['bbox'] # [x1, y1, x2, y2] (이미 안정화됨)
                    })

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = cv2.imread(str(self.frame_dir / item['file_name']))
        if img is None: return None
        
        # --- 1. Crop (이미 안정화된 BBox 사용) ---
        # 주의: DynamicBBoxManager에서 이미 1.2배 확장을 포함하고 있으므로,
        # 여기서 추가로 1.2배를 또 곱하면 안 됨! (중복 확장 방지)
        x1, y1, x2, y2 = item['bbox']
        img_h, img_w = img.shape[:2]
        
        # BBox는 float일 수 있으므로 int 변환
        bw, bh = x2 - x1, y2 - y1
        cx, cy = x1 + bw / 2, y1 + bh / 2
        
        # 이미지 경계 넘지 않도록 Clipping
        nx1, ny1 = max(0, int(x1)), max(0, int(y1))
        nx2, ny2 = min(img_w, int(x2)), min(img_h, int(y2))
        
        crop = img[ny1:ny2, nx1:nx2].copy()
        
        # 만약 crop이 비어있으면(박스가 이미지 밖) 예외처리
        if crop.size == 0: return None, None 

        # --- 2. Gray Padding (Letterbox Resize) ---
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
        input_img = canvas.astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        input_img = (input_img - mean) / std
        input_img = input_img.transpose(2, 0, 1)
        
        # Meta 정보
        meta = {
            'stem': item['stem'],
            'file_name': item['file_name'],
            'frame_idx': item['frame_idx'],
            'obj_id': item['obj_id'],
            'crop_bbox': [nx1, ny1, nx2, ny2],
            'padding': [pad_x, pad_y],
            'scale_factor': scale,
            'input_size': self.input_size
        }
        
        return torch.from_numpy(input_img), meta

def collate_fn(batch):
    batch = [b for b in batch if b is not None and b[0] is not None] # None 필터링 강화
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

    # 🌟 [NEW] BBox 안정화 적용 (Inference 전처리)
    tasks = stabilize_bboxes(tasks)

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
        inputs = inputs.to('cuda', non_blocking=True)

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            feats = model.extract_feat(inputs)
            
            batch_data_samples = [
                PoseDataSample(metainfo=dict(input_size=m['input_size'])) 
                for m in metas
            ]
            preds = model.head.predict(feats, batch_data_samples)

        # 5. 좌표 복원 및 저장
        for i, pred_sample in enumerate(preds):
            meta = metas[i]
            
            if hasattr(pred_sample, 'pred_instances'):
                instances = pred_sample.pred_instances
            else:
                instances = pred_sample
            
            kpts_crop = instances.keypoints
            scores = instances.keypoint_scores
            
            if kpts_crop.ndim == 3:
                kpts_crop = kpts_crop[0]
                scores = scores[0]
            
            pad_x, pad_y = meta['padding']
            scale = meta['scale_factor']
            off_x, off_y = meta['crop_bbox'][:2]
            
            final_kpts = []
            if isinstance(kpts_crop, torch.Tensor): kpts_crop = kpts_crop.cpu().numpy()
            if isinstance(scores, torch.Tensor): scores = scores.cpu().numpy()

            for (cx, cy), score in zip(kpts_crop, scores):
                x_nopad = cx - pad_x
                y_nopad = cy - pad_y
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
        
        del inputs, feats, preds, batch_data_samples # 변수 삭제
        gc.collect()            # 파이썬 가비지 컬렉션
        torch.cuda.empty_cache() # GPU 캐시 비우기 (속도는 약간 느려지지만 안전함)
                
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
    target_df = df[df['n_json'] == 0]

    # 🔍 enumerate를 사용하여 순서(i)와 데이터(index, row)를 분리
    for i, (index, row) in enumerate(target_df.iterrows(), start=1):
        
        print(f"\n================= 처리 시작: {row['common_path']} =================")
        # index+1 대신 i (현재 반복 횟수)를 사용
        print(f"진행상황: {i}/{len(target_df)} (Original Index: {index})") 
        
        COMMON_PATH = row['common_path']
        COMMON_PATH = row['common_path']

        FRAME_DIR = DATA_DIR / "1_FRAME" / COMMON_PATH
        SAM_DIR = DATA_DIR / "8_SAM" / COMMON_PATH
        OUTPUT_DIR = DATA_DIR / "2_KEYPOINTS" / COMMON_PATH

        # COCO 133점 기반    
        # CONFIG = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling/configs/sapiens/sapiens_0.3b-210e_coco_wholebody-1024x768.py"
        # CKPT = DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620.pth"

        # COCO 17점 기반 0.3b
        CONFIG = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling/configs/sapiens/sapiens_0.3b-210e_coco-1024x768.py"
        CKPT = DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_best_coco_AP_796.pth"

        # COCO 17점 기반 0.6b
        # CONFIG = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling/configs/sapiens/sapiens_0.6b-210e_coco-1024x768.py"
        # CKPT = DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.6b_coco_best_coco_AP_812.pth"

        print(f"\noutput_dir: {OUTPUT_DIR}\n")
        # batch_size를 32~64 정도로 높여보세요 (VRAM 허용 시)
        count = run_sapiens_batch_inference(FRAME_DIR, SAM_DIR, OUTPUT_DIR, CONFIG, CKPT, batch_size=50)
        print(f"✅ 완료: {count}개 JSON 생성")

        df.at[index, 'n_json'] = count
        df.to_csv(DATA_DIR / "metadata.csv", index=False)
        print(f"📊 metadata.csv 업데이트 완료")

        MP4_PATH = DATA_DIR / f"3_MP4/{COMMON_PATH}.mp4"

        # 결과 검증 영상 생성
        generate_skeleton_video(
            frame_dir=FRAME_DIR,
            kpt_dir=OUTPUT_DIR,
            output_path=str(MP4_PATH),
            conf_threshold=0
        )