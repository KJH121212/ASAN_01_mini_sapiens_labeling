import os
import sys
import torch
import json
import numpy as np
import cv2
import glob
import time
import shutil
import gc
from collections import OrderedDict
from PIL import Image
from typing import Dict, Any, List, Tuple
from pathlib import Path
from tqdm import tqdm

# --- SAM3 라이브러리 ---
try:
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model_builder import build_sam3_video_model
    from sam3.train.data.collator import collate_fn_api as collate
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
    from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
    from sam3.eval.postprocessors import PostProcessImage
    
    # SAM3 Root Path Fix
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    sys.path.append(f"{sam3_root}/examples")
except ImportError:
    print("❌ [Error] SAM3 라이브러리를 찾을 수 없습니다.")

# --- MMPose (Sapiens) 라이브러리 ---
try:
    from mmpose.apis import init_model
    from mmpose.utils import register_all_modules
    from mmpose.structures import PoseDataSample
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("❌ [Error] MMPose 라이브러리가 필요합니다.")

# ==============================================================================
# [Config] 전체 설정 (경로 및 파라미터)
# ==============================================================================
class Config:
    # 1. 공통 경로 설정
    BASE_DIR = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data"
    COMMON_PATH = "AI_dataset/N02/N02_Treatment/diagonal__biceps_curl"
    
    # 입력/출력 디렉토리
    FRAME_DIR = os.path.join(BASE_DIR, "1_FRAME", COMMON_PATH)
    SAM_OUTPUT_DIR = os.path.join(BASE_DIR, "8_SAM_Integrated", COMMON_PATH) # 중간 결과(BBox) 저장소
    FINAL_OUTPUT_DIR = os.path.join(BASE_DIR, "9_KEYPOINTS_V2", COMMON_PATH) # 최종 결과(Keypoints) 저장소
    
    # 2. SAM3 설정
    SAM_CHECKPOINT = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/SAM3/sam3.pt"
    SAM_VOCAB = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/SAM3/bpe_simple_vocab_16e6.txt.gz"
    PROMPT = "person"
    START_FRAME_IDX = 0  # 트래킹 시작 프레임
    
    # 3. Sapiens 설정
    SAPIENS_CONFIG = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling/configs/sapiens/sapiens_0.6b-210e_coco-1024x768.py"
    SAPIENS_CKPT = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/sapiens/pose/sapiens_0.6b_coco_best_coco_AP_812.pth"
    BATCH_SIZE = 16  # VRAM 상황에 맞춰 조절

    # 시스템 설정
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
GLOBAL_COUNTER = 1

# ==============================================================================
# [Utils] 공통 유틸리티
# ==============================================================================
def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def clear_gpu_memory():
    """GPU 메모리 강제 정리"""
    print("🧹 [System] GPU 메모리 정리 중...")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print(f"   Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

def mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return {"size": mask.shape, "counts": runs.tolist()}

def mask_to_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols): return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

# ==============================================================================
# [Stage 1] SAM3 Tracking Logic
# ==============================================================================
def create_empty_datapoint(): return Datapoint(find_queries=[], images=[])

def set_image(datapoint, pil_image):
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]

def add_text_prompt(datapoint, text_query):
    global GLOBAL_COUNTER
    w, h = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query, image_id=0, object_ids_output=[], is_exhaustive=True, query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_COUNTER, original_image_id=GLOBAL_COUNTER, original_category_id=1,
                original_size=[w, h], object_id=0, frame_index=0,
            )
        )
    )
    GLOBAL_COUNTER += 1

def detect_objects_sam3(frame_dir: str, text_prompt: str, target_frame_idx: int) -> Dict[str, Any]:
    print(f"--- [SAM3] 객체 검출 (Frame: {target_frame_idx}, Prompt: '{text_prompt}') ---")
    candidates = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")) + glob.glob(os.path.join(frame_dir, "*.png")))
    try: candidates.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    except: candidates.sort()

    if not candidates: return None
    img_path = candidates[target_frame_idx]
    
    model = build_sam3_image_model(checkpoint_path=cfg.SAM_CHECKPOINT, bpe_path=cfg.SAM_VOCAB)
    model.to(cfg.DEVICE)

    transform = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(), NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    postprocessor = PostProcessImage(
        max_dets_per_img=-1, iou_type="segm", use_original_sizes_box=True, use_original_sizes_mask=True,
        convert_mask_to_rle=False, detection_threshold=0.5, to_cpu=False
    )

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        datapoint = create_empty_datapoint()
        set_image(datapoint, Image.open(img_path).convert("RGB"))
        add_text_prompt(datapoint, text_prompt)
        datapoint = transform(datapoint)
        batch = collate([datapoint], dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, cfg.DEVICE, non_blocking=True)
        processed_results = postprocessor.process_results(model(batch), batch.find_metadatas)

    del model; del batch
    clear_gpu_memory()

    if len(processed_results) > 0:
        return list(processed_results.values())[0]
    return None

class LazyVideoLoader:
    def __init__(self, video_path, image_size=1008):
        self.frame_paths = sorted(glob.glob(os.path.join(video_path, "*.jpg")) + 
                                  glob.glob(os.path.join(video_path, "*.png")))
        try: self.frame_paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        except: self.frame_paths.sort()
        self.image_size = image_size
    def __len__(self): return len(self.frame_paths)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.frame_paths[idx]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = (img.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return torch.from_numpy(img).permute(2, 0, 1)

def save_frame_json(frame_idx, obj_ids, video_res_masks, state, output_dir):
    frame_data = {
        "frame_index": frame_idx,
        "file_name": os.path.basename(state["images"].frame_paths[frame_idx]),
        "objects": []
    }
    if video_res_masks is not None and len(video_res_masks) > 0:
        for k, obj_id in enumerate(obj_ids):
            if isinstance(obj_id, torch.Tensor): obj_id = obj_id.item()
            mask_tensor = video_res_masks[k]
            if mask_tensor.dim() == 3: mask_tensor = mask_tensor.squeeze(0)
            mask_np = (mask_tensor.cpu().numpy() > 0.0).astype(np.uint8)
            if np.any(mask_np):
                frame_data["objects"].append({
                    "id": obj_id,
                    "segmentation": mask_to_rle(mask_np),
                    "bbox": mask_to_bbox(mask_np)
                })
    json_path = os.path.join(output_dir, f"{frame_idx:06d}.json")
    with open(json_path, 'w') as f: json.dump(frame_data, f)

def run_sam3_tracking_stage(detection_results):
    print(f"--- [SAM3] 양방향 트래킹 시작 ---")
    os.makedirs(cfg.SAM_OUTPUT_DIR, exist_ok=True)
    
    sam3_model = build_sam3_video_model(apply_temporal_disambiguation=True, device="cuda")
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    
    # Lazy Init State
    state = {
        "offload_video_to_cpu": True, "offload_state_to_cpu": True,
        "device": predictor.device, "storage_device": torch.device("cpu"),
        "images": LazyVideoLoader(cfg.FRAME_DIR, predictor.image_size),
        "point_inputs_per_obj": {}, "mask_inputs_per_obj": {}, "cached_features": {}, "constants": {},
        "obj_id_to_idx": OrderedDict(), "obj_idx_to_id": OrderedDict(), "obj_ids": [],
        "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        "tracking_has_started": False, "frames_already_tracked": {},
        "first_ann_frame_idx": None, "output_dict_per_obj": {}, "temp_output_dict_per_obj": {},
        "consolidated_frame_inds": {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()}
    }
    state["num_frames"] = len(state["images"])
    first = cv2.imread(state["images"].frame_paths[0])
    state["video_height"], state["video_width"] = first.shape[:2]
    predictor.clear_all_points_in_video(state)

    # Initial Mask Registration
    mask_key = "masks" if "masks" in detection_results else "segmentation"
    for i in range(detection_results["scores"].numel()):
        mask = detection_results[mask_key][i].cuda().float()
        if mask.dim() == 3: mask = mask.squeeze(0)
        predictor.add_new_mask(inference_state=state, frame_idx=cfg.START_FRAME_IDX, obj_id=i+1, mask=mask)

    # Forward
    print(f"➡️ 정방향 추적 ({cfg.START_FRAME_IDX} -> End)")
    for frame_idx, obj_ids, _, video_res_masks, _ in predictor.propagate_in_video(state, start_frame_idx=cfg.START_FRAME_IDX, reverse=False):
        save_frame_json(frame_idx, obj_ids, video_res_masks, state, cfg.SAM_OUTPUT_DIR)
        # VRAM Management (Simple)
        if frame_idx > cfg.START_FRAME_IDX and frame_idx % 100 == 0:
            torch.cuda.empty_cache()

    # Clear Forward Memory
    print("🧹 방향 전환 전 메모리 정리...")
    outputs = state["output_dict"]["non_cond_frame_outputs"]
    keys_to_remove = [k for k in outputs.keys() if k > cfg.START_FRAME_IDX]
    for k in keys_to_remove: del outputs[k]
    torch.cuda.empty_cache()

    # Backward
    print(f"⬅️ 역방향 추적 ({cfg.START_FRAME_IDX} -> 0)")
    for frame_idx, obj_ids, _, video_res_masks, _ in predictor.propagate_in_video(state, start_frame_idx=cfg.START_FRAME_IDX, reverse=True):
        save_frame_json(frame_idx, obj_ids, video_res_masks, state, cfg.SAM_OUTPUT_DIR)
        if frame_idx < cfg.START_FRAME_IDX and frame_idx % 100 == 0:
            torch.cuda.empty_cache()

    del predictor; del sam3_model; del state
    clear_gpu_memory()

# ==============================================================================
# [Stage 2] Sapiens Pose Estimation Logic
# ==============================================================================
class DynamicBBoxManager:
    def __init__(self, expand_ratio=1.0, min_occupancy=0.7, smoothing=0.2, cooldown=30):
        self.expand_ratio = expand_ratio
        self.margin_threshold = 1.0 - min_occupancy 
        self.alpha = smoothing
        self.current_box = None 
        self.cooldown_max = cooldown
        self.cooldown_timer = 0

    def update(self, mask_bbox):
        if self.current_box is None:
            self.current_box = self._make_target_box(mask_bbox, self.expand_ratio)
            return self.current_box
        if self.cooldown_timer > 0: self.cooldown_timer -= 1

        if self._is_out_of_bound(mask_bbox, self.current_box):
            target_box = self._make_target_box(mask_bbox, self.expand_ratio)
            self.current_box = self._smooth_update(self.current_box, target_box)
            self.cooldown_timer = self.cooldown_max
        elif self.cooldown_timer == 0 and self._has_excessive_margin(mask_bbox, self.current_box, self.margin_threshold):
            target_box = self._make_target_box(mask_bbox, self.expand_ratio)
            self.current_box = self._smooth_update(self.current_box, target_box)
            self.cooldown_timer = self.cooldown_max
        return self.current_box

    def _make_target_box(self, box, ratio):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        nw, nh = w * ratio, h * ratio
        return [cx - nw/2, cy - nh/2, cx + nw/2, cy + nh/2]
    
    def _is_out_of_bound(self, mask, view):
        pad = 0.1
        return (mask[0] < view[0]-pad) or (mask[1] < view[1]-pad) or (mask[2] > view[2]+pad) or (mask[3] > view[3]+pad)
    
    def _has_excessive_margin(self, mask, view, threshold):
        view_w, view_h = view[2] - view[0], view[3] - view[1]
        if view_w <= 0 or view_h <= 0: return True
        return (1.0 - ((mask[2]-mask[0])/view_w) > threshold) or (1.0 - ((mask[3]-mask[1])/view_h) > threshold)
    
    def _smooth_update(self, current, target):
        return [c * (1 - self.alpha) + t * self.alpha for c, t in zip(current, target)]

def stabilize_bboxes(tasks):
    print("🌊 [Sapiens] BBox 안정화 로직 적용 중 (Hysteresis)...")
    managers = {} 
    for sam_file, file_name, objects in tasks:
        for obj in objects:
            if not obj.get('bbox'): continue
            oid = obj['id']
            if oid not in managers:
                managers[oid] = DynamicBBoxManager(expand_ratio=1.1, min_occupancy=0.7, smoothing=0.2)
            obj['bbox'] = managers[oid].update(obj['bbox'])
    return tasks

class SapiensBatchDataset(Dataset):
    def __init__(self, tasks, frame_dir, input_size=(1024, 768)):
        self.frame_dir = Path(frame_dir)
        self.items = []
        self.input_size = input_size
        for sam_file, file_name, objects in tasks:
            f_idx = int(sam_file.stem) if sam_file.stem.isdigit() else 0
            for obj in objects:
                if obj.get('bbox'):
                    self.items.append({
                        'stem': sam_file.stem, 'file_name': file_name, 'frame_idx': f_idx, 
                        'obj_id': obj['id'], 'bbox': obj['bbox']
                    })

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = cv2.imread(str(self.frame_dir / item['file_name']))
        if img is None: return None
        
        x1, y1, x2, y2 = item['bbox']
        img_h, img_w = img.shape[:2]
        nx1, ny1 = max(0, int(x1)), max(0, int(y1))
        nx2, ny2 = min(img_w, int(x2)), min(img_h, int(y2))
        crop = img[ny1:ny2, nx1:nx2].copy()
        if crop.size == 0: return None, None 

        target_w, target_h = self.input_size
        h, w = crop.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(crop, (new_w, new_h))
        canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
        pad_x, pad_y = (target_w - new_w) // 2, (target_h - new_h) // 2
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        input_img = (canvas.astype(np.float32) - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
        input_img = input_img.transpose(2, 0, 1)
        
        meta = {
            'stem': item['stem'], 'file_name': item['file_name'], 'frame_idx': item['frame_idx'],
            'obj_id': item['obj_id'], 'crop_bbox': [nx1, ny1, nx2, ny2],
            'padding': [pad_x, pad_y], 'scale_factor': scale, 'input_size': self.input_size
        }
        return torch.from_numpy(input_img), meta

def collate_fn(batch):
    batch = [b for b in batch if b is not None and b[0] is not None]
    if not batch: return None
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]

def to_py(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, dict): return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_py(v) for v in obj]
    return obj

def run_sapiens_estimation_stage():
    print("--- [Sapiens] Pose Estimation 시작 ---")
    
    # 1. Load SAM JSONs (Replace external function)
    sam_dir = Path(cfg.SAM_OUTPUT_DIR)
    sam_files = sorted(list(sam_dir.glob("*.json")))
    tasks = []
    print("📂 SAM JSON 스캔 및 파싱...")
    
    for sam_file in tqdm(sam_files):
        with open(sam_file, 'r') as f:
            data = json.load(f)
            file_name = data.get('file_name', '')
            objects = data.get('objects', [])
            if (Path(cfg.FRAME_DIR) / file_name).exists():
                tasks.append((sam_file, file_name, objects))

    # 2. BBox Stabilization
    tasks = stabilize_bboxes(tasks)

    # 3. Model Init
    print("🚀 Sapiens 모델 로드...")
    register_all_modules()
    model = init_model(cfg.SAPIENS_CONFIG, cfg.SAPIENS_CKPT, device='cuda:0')
    model.eval()

    # 4. DataLoader
    dataset = SapiensBatchDataset(tasks, cfg.FRAME_DIR, input_size=(1024, 768))
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, num_workers=4, shuffle=False, collate_fn=collate_fn)

    # 5. Output Dir Prep
    output_dir = Path(cfg.FINAL_OUTPUT_DIR)
    if output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"⚡ Batch Inference 수행 (Objects: {len(dataset)})")
    
    for batch in tqdm(loader, desc="Sapiens Inference"):
        if batch is None: continue
        inputs, metas = batch
        inputs = inputs.to('cuda', non_blocking=True)

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            feats = model.extract_feat(inputs)
            batch_data_samples = [PoseDataSample(metainfo=dict(input_size=m['input_size'])) for m in metas]
            preds = model.head.predict(feats, batch_data_samples)

        for i, pred_sample in enumerate(preds):
            meta = metas[i]
            instances = getattr(pred_sample, 'pred_instances', pred_sample)
            kpts_crop = instances.keypoints
            scores = instances.keypoint_scores
            
            if kpts_crop.ndim == 3: kpts_crop, scores = kpts_crop[0], scores[0]
            
            pad_x, pad_y = meta['padding']
            scale = meta['scale_factor']
            off_x, off_y = meta['crop_bbox'][:2]
            
            final_kpts = []
            if isinstance(kpts_crop, torch.Tensor): kpts_crop = kpts_crop.cpu().numpy()
            if isinstance(scores, torch.Tensor): scores = scores.cpu().numpy()

            for (cx, cy), score in zip(kpts_crop, scores):
                fx = ((cx - pad_x) / scale) + off_x
                fy = ((cy - pad_y) / scale) + off_y
                final_kpts.append([float(fx), float(fy), float(score)])
            
            instance_item = {
                "instance_id": int(meta['obj_id']),
                "keypoints": final_kpts,
                "keypoint_scores": scores.tolist(),
                "bbox": [float(v) for v in meta['crop_bbox']]
            }
            
            save_path = output_dir / f"{meta['stem']}.json"
            if save_path.exists():
                with open(save_path, "r", encoding="utf-8") as f: data_j = json.load(f)
                data_j['instance_info'].append(instance_item)
            else:
                data_j = {
                    "frame_index": meta['frame_idx'], "file_name": meta['file_name'],
                    "meta_info": to_py(model.dataset_meta), "instance_info": [instance_item]
                }
            with open(save_path, "w", encoding="utf-8") as f: json.dump(data_j, f, ensure_ascii=False, indent=2)

    del model; del loader; del dataset
    clear_gpu_memory()

# ==============================================================================
# [Main] 통합 실행
# ==============================================================================
def main():
    total_start = time.time()
    
    # 0. 경로 확인
    if not os.path.exists(cfg.FRAME_DIR):
        print(f"❌ 입력 디렉토리가 없습니다: {cfg.FRAME_DIR}")
        return

    # 1. SAM3 실행
    print("="*60)
    print("🚀 [Step 1/3] SAM3 Object Tracking & BBox Generation")
    print("="*60)
    
    sam_start = time.time()
    detection_res = detect_objects_sam3(cfg.FRAME_DIR, cfg.PROMPT, cfg.START_FRAME_IDX)
    
    if detection_res:
        run_sam3_tracking_stage(detection_results=detection_res)
        print(f"✅ SAM3 완료. 소요시간: {(time.time() - sam_start)/60:.1f}분")
    else:
        print("❌ 객체 검출 실패. 종료합니다.")
        return

    # 2. 메모리 정리 (중요)
    print("\n" + "="*60)
    print("🧹 [Step 2/3] Switching Models (Clearing VRAM)")
    print("="*60)
    clear_gpu_memory()
    time.sleep(2) # 쿨다운

    # 3. Sapiens 실행
    print("\n" + "="*60)
    print("🚀 [Step 3/3] Sapiens Pose Estimation")
    print("="*60)
    
    sapiens_start = time.time()
    run_sapiens_estimation_stage()
    print(f"✅ Sapiens 완료. 소요시간: {(time.time() - sapiens_start)/60:.1f}분")

    # Final Report
    total_elapsed = time.time() - total_start
    print("\n" + "="*60)
    print(f"🎉 모든 작업 완료! 총 소요시간: {total_elapsed/60:.1f}분")
    print(f"📁 최종 결과 경로: {cfg.FINAL_OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    force_cudnn_initialization()
    main()