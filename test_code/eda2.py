import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def analyze_score_auto_scale(input_dir, kpt_idx, output_dir):
    input_path = Path(input_dir)
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. JSON 로딩
    json_files = sorted(list(input_path.glob("*.json")), key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem)
    if not json_files: return

    frames = []
    scores = []

    print(f"📂 [{kpt_idx}번 Keypoint] 데이터 로딩 중... ({len(json_files)} frames)")

    for f_path in tqdm(json_files):
        with open(f_path, 'r') as f:
            data = json.load(f)
        if not data.get('instance_info'): continue
        kpts = data['instance_info'][0]['keypoints']
        
        if kpt_idx < len(kpts):
            s = kpts[kpt_idx][2]
            frames.append(int(f_path.stem) if f_path.stem.isdigit() else len(frames))
            scores.append(s)

    scores = np.array(scores)
    frames = np.array(frames)

    # 2. 통계 및 범위 계산 (Auto Scaling)
    if len(scores) == 0: return

    mean_score = np.mean(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    
    # 그래프 범위를 데이터의 min/max에 약간의 여유를 둬서 설정
    y_min = max(0, min_score - (max_score - min_score) * 0.1)
    y_max = max_score + (max_score - min_score) * 0.1
    
    # 만약 모든 값이 0이면 강제로 0~0.01 설정
    if y_max == 0: y_max = 0.01

    print(f"📊 Auto-Scaling 적용: {min_score:.6f} ~ {max_score:.6f}")

    # 3. 시각화
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Confidence Score Analysis (Auto-Scaled): ID {kpt_idx}", fontsize=16, fontweight='bold')

    # [좌측] Histogram (데이터 범위에 맞춤)
    # bins를 0~1 고정이 아니라, 실제 데이터 범위(min~max)로 설정
    axes[0].hist(scores, bins=50, color='royalblue', edgecolor='black', alpha=0.7)
    axes[0].set_title(f"Score Distribution ({min_score:.4f}~{max_score:.4f})", fontsize=14)
    axes[0].set_xlabel("Confidence Score")
    axes[0].set_ylabel("Count")
    axes[0].axvline(mean_score, color='red', linestyle='--', label=f'Mean: {mean_score:.4f}')
    axes[0].legend()

    # [우측] Timeline (Y축 자동 확대)
    axes[1].plot(frames, scores, color='darkviolet', linewidth=1, alpha=0.8)
    axes[1].set_title(f"Score Trend (Zoomed In)", fontsize=14)
    axes[1].set_xlabel("Frame Index")
    axes[1].set_ylabel("Score")
    
    # 🔴 여기가 핵심: Y축 범위를 데이터에 맞춤
    axes[1].set_ylim(y_min, y_max)
    
    # 평균선 표시
    axes[1].axhline(mean_score, color='red', linestyle='--', alpha=0.5, label='Mean')
    axes[1].legend()

    plt.tight_layout()

    # 저장
    file_name = f"eda_score_auto_kpt{kpt_idx:03d}.png"
    out_file = save_path / file_name
    plt.savefig(out_file, dpi=150)
    plt.close()
    
    print(f"✅ 저장 완료: {out_file}")

# ============================================================
if __name__ == "__main__":
    DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
    df = pd.read_csv(DATA_DIR / "metadata.csv")
    
    target = 4
    COMMON_PATH = df['common_path'][target]
    INPUT_DIR = DATA_DIR / "9_KEYPOINTS_V2" / COMMON_PATH
    OUTPUT_DIR = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling/eda_results"
    
    TARGET_KEYPOINT = 6
    analyze_score_auto_scale(INPUT_DIR, TARGET_KEYPOINT, OUTPUT_DIR)