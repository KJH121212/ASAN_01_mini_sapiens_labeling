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

# --- MMPose 및 MMEngine 라이브러리 임포트 ---
# Sapiens 모델은 MMPose 프레임워크를 기반으로 동작하므로 관련 모듈이 필수입니다.
try:
    from mmpose.apis import init_model
    from mmpose.utils import register_all_modules
    from mmpose.structures import split_instances, merge_data_samples
    from mmengine.dataset import pseudo_collate
except ImportError:
    print("❌ MMPose 라이브러리가 필요합니다. (pip install mmpose mmengine)")
    sys.exit(1)

# --- 사용자 정의 함수 임포트 ---
# SAM 결과 JSON 파일에서 BBox(사람 위치)와 ID를 추출하는 함수입니다.
try:
    from functions.extract_bbox_and_id import extract_bbox_and_id
except ImportError:
    # 경로가 안 맞을 경우 상위 폴더를 참조하도록 설정
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from functions.extract_bbox_and_id import extract_bbox_and_id

# --- JSON 직렬화 헬퍼 함수 ---
# NumPy 배열이나 float32 같은 비-표준 타입을 JSON 저장 가능한 기본 타입(float, int, list)으로 변환합니다.
def to_py(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, dict): return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_py(v) for v in obj]
    return obj

# ==================================================================================
# 1️⃣ Dataset 정의: SAM BBox를 기반으로 이미지를 잘라내어(Crop) 모델 입력으로 변환
# ==================================================================================
class SapiensLiteDataset(Dataset):
    """
    Sapiens 모델 추론을 위한 데이터셋 클래스
    - SAM이 찾은 사람 영역(BBox)을 기준으로 이미지를 잘라냅니다(Crop).
    - 잘라낸 이미지를 모델 입력 크기(1024x768)로 변환(Resize) 및 정규화(Normalize)합니다.
    """
    def __init__(self, tasks, frame_dir):
        self.frame_dir = Path(frame_dir)
        self.items = []
        self.input_res = (1024, 768)  # Sapiens 모델의 고정 입력 해상도 (H, W)
        
        # tasks: (SAM결과파일, 이미지파일명, 객체리스트) 튜플의 리스트
        for sam_file, file_name, objects in tasks:
            # 파일명에서 프레임 번호 추출 (예: 000123.json -> 123)
            f_idx = int(sam_file.stem) if sam_file.stem.isdigit() else 0
            
            for obj in objects:
                # BBox가 있는 유효한 객체만 처리
                if obj.get('bbox'):
                    self.items.append({
                        'stem': sam_file.stem,       # 파일 식별자
                        'file_name': file_name,      # 이미지 파일명
                        'frame_idx': f_idx,          # 프레임 인덱스
                        'obj_id': obj['id'],         # 객체 ID (Tracking 결과)
                        'bbox': obj['bbox']          # SAM이 찾은 원본 BBox [x1, y1, x2, y2]
                    })
                    
    def __len__(self): 
        return len(self.items)
        
    def __getitem__(self, idx):
        item = self.items[idx]
        
        # 1. 이미지 로드
        img = cv2.imread(str(self.frame_dir / item['file_name']))
        if img is None: return None
        
        x1, y1, x2, y2 = item['bbox']
        h, w = img.shape[:2]
        
        # 2. Crop 영역 계산 (Top-Down 방식의 핵심)
        # 사람 영역을 너무 타이트하게 자르면 포즈 추정이 어려우므로 1.2배 확장합니다.
        bw, bh = x2 - x1, y2 - y1       # 박스 너비, 높이
        cx, cy = x1 + bw / 2, y1 + bh / 2 # 박스 중심점
        
        nw, nh = bw * 1.2, bh * 1.2     # 1.2배 확장된 크기
        
        # 이미지 경계를 벗어나지 않도록 좌표 보정
        nx1, ny1 = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
        nx2, ny2 = min(w, int(cx + nw / 2)), min(h, int(cy + nh / 2))
        
        # 3. 이미지 잘라내기 (Crop)
        crop = img[ny1:ny2, nx1:nx2].copy()
        
        # 4. 모델 입력 크기로 리사이즈 (Resize)
        # Sapiens는 (1024, 768) 입력을 기대합니다. 비율이 달라도 강제로 맞춥니다.
        input_img = cv2.resize(crop, (self.input_res[1], self.input_res[0]))
        
        # 5. 전처리 (HWC -> CHW, 정규화)
        input_img = input_img.transpose(2, 0, 1).astype(np.float32)
        
        # ImageNet 평균/표준편차 정규화 값
        mean = np.array([123.675, 116.28, 103.53]).reshape(3, 1, 1).astype(np.float32)
        std = np.array([58.395, 57.12, 57.375]).reshape(3, 1, 1).astype(np.float32)
        input_img = (input_img - mean) / std
        
        # 6. 결과 반환 (텐서, 메타데이터)
        # 메타데이터는 나중에 모델 출력 좌표를 원본 이미지 좌표로 복구할 때 사용됩니다.
        return torch.from_numpy(input_img), {
            'stem': item['stem'], 
            'file_name': item['file_name'],
            'frame_idx': item['frame_idx'], 
            'obj_id': item['obj_id'],
            # 좌표 복구를 위한 오프셋(Crop 시작점)과 스케일(Resize 비율) 정보
            'offset': [nx1, ny1], 
            'scale': [crop.shape[1] / self.input_res[1], crop.shape[0] / self.input_res[0]]
        }

# DataLoader에서 None 데이터를 필터링하고 배치 구성을 도와주는 함수
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    # (Input Tensors, Metadata Lists) 형태로 분리하여 반환
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]

# ==================================================================================
# 2️⃣ 메인 추론 함수: 모델 로드 -> 추론 -> 좌표 복원 -> JSON 저장
# ==================================================================================
def run_sapiens_lite_inference(frame_dir, sam_dir, output_dir, config_path, ckpt_path, batch_size=25):
    frame_dir, sam_dir, output_dir = Path(frame_dir), Path(sam_dir), Path(output_dir)
    
    # 출력 폴더 초기화 (기존 결과 삭제 후 재생성)
    if output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    register_all_modules() # MMPose 모듈 등록
    
    # 1. SAM JSON 파일 리스트업 및 작업 목록 생성
    sam_files = sorted(list(sam_dir.glob("*.json")))
    tasks = []
    print("🔍 SAM JSON 스캔 중...")
    for sam_file in tqdm(sam_files):
        file_name, objects = extract_bbox_and_id(str(sam_file))
        # 해당 프레임 이미지가 실제로 존재할 때만 작업 목록에 추가
        if (frame_dir / file_name).exists():
            tasks.append((sam_file, file_name, objects))

    # 2. Sapiens 모델 로드 (GPU 사용)
    print("🚀 Sapiens 모델 로드 중...")
    model = init_model(str(config_path), str(ckpt_path), device='cuda:0')
    model.eval()

    # 3. 데이터셋 및 데이터로더 생성
    dataset = SapiensLiteDataset(tasks, frame_dir)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False, collate_fn=collate_fn, pin_memory=True)

    print(f"🔥 Sapiens-Lite 고속 추론 및 Full 포맷 저장 시작")
    
    # 4. 배치 단위 추론 반복
    for batch in tqdm(loader, desc="Processing"):
        if batch is None: continue
        inputs, metas = batch
        inputs = inputs.to('cuda', non_blocking=True) # 입력 데이터를 GPU로 이동

        with torch.no_grad():
            # [4-1] 모델 Forward
            features = model.backbone(inputs)
            
            # Heatmap Head를 통해 관절별 확률지도(Heatmap) 생성
            # 출력 형태: (Batch, Num_Keypoints, Height/4, Width/4)
            heatmaps = model.head(features)
            if isinstance(heatmaps, (list, tuple)): heatmaps = heatmaps[-1]

            # [4-2] 좌표 디코딩 (Heatmap -> 좌표)
            # 가장 높은 확률을 가진 픽셀 위치(argmax)를 찾습니다.
            B, C, H, W = heatmaps.shape
            heatmaps_reshaped = heatmaps.view(B, C, -1)
            max_vals, max_idxs = torch.max(heatmaps_reshaped, dim=2)
            
            # 모델 출력은 입력 해상도의 1/4 크기이므로 4를 곱해줍니다.
            preds_x = (max_idxs % W).float() * 4 
            preds_y = (max_idxs // W).float() * 4

            # [4-3] 배치 내 각 샘플 결과 처리
            for i in range(B):
                meta = metas[i]
                
                # --- 좌표 원복 (중요) ---
                # 모델이 본 좌표(Resize된 Crop 이미지 기준)를 원본 이미지(Full Frame) 좌표로 변환
                # 공식: (모델좌표 * 스케일) + 오프셋
                final_x = (preds_x[i].cpu().numpy() * meta['scale'][0]) + meta['offset'][0]
                final_y = (preds_y[i].cpu().numpy() * meta['scale'][1]) + meta['offset'][1]
                scores = max_vals[i].cpu().numpy() # 신뢰도 점수
                
                # [x, y, score] 형태로 결합
                keypoints_full = np.stack([final_x, final_y, scores], axis=1).tolist()
                
                # --- 결과 저장용 객체 생성 ---
                instance_item = {
                    "instance_id": int(meta['obj_id']),
                    "keypoints": keypoints_full,
                    "keypoint_scores": scores.tolist(),
                    # 참고: 여기서 bbox는 실제 모델 추론에 사용된 'Crop 영역'을 역산한 값입니다.
                    # SAM 원본 bbox와 미세하게 다를 수 있습니다. (원본 유지가 필요하면 수정 필요)
                    "bbox": [ 
                        meta['offset'][0], meta['offset'][1], 
                        meta['offset'][0] + (1024 * meta['scale'][0]), 
                        meta['offset'][1] + (768 * meta['scale'][1])
                    ]
                }

                # --- JSON 파일 쓰기 ---
                save_path = output_dir / f"{meta['stem']}.json"
                
                # 기존 파일이 있으면 로드해서 append(다중 객체인 경우), 없으면 새로 생성
                if save_path.exists():
                    with open(save_path, "r") as f: data_j = json.load(f)
                    data_j['instance_info'].append(instance_item)
                else:
                    data_j = {
                        "frame_index": meta['frame_idx'],
                        "file_name": meta['file_name'],
                        "meta_info": to_py(model.dataset_meta), # Keypoint 정의 등 모델 메타정보
                        "instance_info": [instance_item]
                    }
                
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(data_j, f, ensure_ascii=False, indent=2)

        # 메모리 정리
        torch.cuda.empty_cache()

    return len(list(output_dir.glob("*.json")))

# ==================================================================================
# 3️⃣ 실행 진입점 (Main)
# ==================================================================================
if __name__ == "__main__":
    # 데이터 경로 설정
    DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
    
    # 메타데이터 파일 로드 및 처리할 비디오 경로 선택
    df = pd.read_csv(DATA_DIR / "metadata.csv")
    COMMON_PATH = df['common_path'][1] # 예시로 두 번째 비디오 선택
    
    # 입출력 디렉토리 설정
    FRAME_DIR = DATA_DIR / "1_FRAME" / COMMON_PATH       # 원본 이미지 폴더
    SAM_DIR = DATA_DIR / "8_SAM" / COMMON_PATH           # SAM 결과(bbox) 폴더
    OUTPUT_DIR = DATA_DIR / "9_KEYPOINTS_V2" / COMMON_PATH # 최종 Pose 결과 저장 폴더
    
    # 모델 설정 파일 및 체크포인트 경로
    CONFIG = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling/configs/sapiens/sapiens_0.3b-210e_coco_wholebody-1024x768.py"
    CKPT = DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620.pth"

    # 추론 실행 (Batch Size 조절 가능)
    count = run_sapiens_lite_inference(FRAME_DIR, SAM_DIR, OUTPUT_DIR, CONFIG, CKPT, batch_size=30)
    print(f"✅ 완료: {count}개 JSON 생성")