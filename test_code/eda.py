import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def analyze_keypoint_trajectory(kpt_path, kpt_idx, out_path):
    """
    Args:
        kpt_path (str): JSON 폴더 경로
        kpt_idx (int): 분석할 키포인트 인덱스 (숫자만 입력)
        out_path (str): 저장할 폴더
    """
    kpt_dir = Path(kpt_path)
    save_dir = Path(out_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 입력값 정수 변환 (혹시 문자열 숫자가 들어올 경우 대비)
    try:
        kpt_idx = int(kpt_idx)
    except ValueError:
        print("❌ [Error] 키포인트 ID는 반드시 '숫자'여야 합니다.")
        return

    print(f"🎯 분석 시작: Keypoint ID [{kpt_idx}]")

    # 1. JSON 파일 로딩
    json_files = sorted(list(kpt_dir.glob("*.json")), key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem)
    if not json_files:
        print("❌ JSON 파일이 없습니다.")
        return

    frames, x_coords, y_coords = [], [], []

    # 2. 데이터 추출
    print(f"📂 데이터 스캔 중... ({len(json_files)} frames)")
    for f_path in tqdm(json_files):
        with open(f_path, 'r') as f:
            data = json.load(f)
        
        # 첫 번째 사람만 분석
        if not data.get('instance_info'): continue
        
        kpts = data['instance_info'][0]['keypoints']
        
        # 인덱스 범위 체크
        if kpt_idx < len(kpts):
            x, y, s = kpts[kpt_idx]
            
            # 프레임 번호 및 좌표 저장
            frames.append(int(f_path.stem) if f_path.stem.isdigit() else len(frames))
            x_coords.append(x)
            y_coords.append(y)
        else:
            # 해당 프레임에 요청한 ID의 키포인트가 없는 경우 (데이터셋 불일치 등)
            pass

    if not frames:
        print(f"❌ ID {kpt_idx}에 해당하는 데이터가 하나도 없습니다.")
        return

    # 3. Numpy 변환 및 이동 거리 계산
    frames = np.array(frames)
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    
    # 유클리드 거리 (Velocity)
    dx = np.diff(x_coords, prepend=x_coords[0])
    dy = np.diff(y_coords, prepend=y_coords[0])
    distances = np.sqrt(dx**2 + dy**2)

    # 4. 시각화 (3-Panel Plot)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle(f"Trajectory Analysis: Keypoint ID {kpt_idx}", fontsize=18, fontweight='bold')

    # (1) X 좌표
    axes[0].plot(frames, x_coords, color='royalblue', linewidth=1.2)
    axes[0].set_title(f"X Coordinate", fontsize=14)
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Pixel X")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # (2) Y 좌표 (Inverted)
    axes[1].plot(frames, y_coords, color='forestgreen', linewidth=1.2)
    axes[1].set_title(f"Y Coordinate (Inverted)", fontsize=14)
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Pixel Y")
    axes[1].invert_yaxis() # 이미지 좌표계 반영
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # (3) 이동 거리 (Euclidean Distance)
    axes[2].plot(frames, distances, color='crimson', linewidth=1.2, label='Movement')
    
    # 평균선 및 Jitter 표시
    mean_dist = np.mean(distances)
    axes[2].axhline(mean_dist, color='black', linestyle='--', alpha=0.7, label=f'Mean: {mean_dist:.2f}')
    
    jitter_threshold = mean_dist + 3 * np.std(distances)
    outliers = np.where(distances > jitter_threshold)[0]
    if len(outliers) > 0:
        axes[2].scatter(frames[outliers], distances[outliers], color='black', s=20, zorder=5, label='Potential Jitter')

    axes[2].set_title(f"Frame-to-Frame Velocity", fontsize=14)
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Pixel Distance")
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    # 저장 (파일명에 ID 포함)
    save_filename = f"eda_kpt_{kpt_idx:03d}.png"
    save_path_full = save_dir / save_filename
    plt.savefig(save_path_full, dpi=150)
    plt.close()
    
    print(f"✅ 그래프 저장 완료: {save_path_full}")

# ============================================================
# ▶️ 사용법
# ============================================================
if __name__ == "__main__":
    DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
    df = pd.read_csv(DATA_DIR / "metadata.csv")
    target = 4
    COMMON_PATH = df['common_path'][target]
    INPUT_DIR = DATA_DIR / "9_KEYPOINTS_V2" / COMMON_PATH

    OUTPUT_DIR = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling/eda_results"
    
    # 3. 분석할 키포인트: '번호(int)' 또는 '이름(str)' 모두 가능
    # 예: 오른쪽 손목(10), 오른쪽 발목(16), 오른쪽 검지 끝(120) 등
    TARGET_KEYPOINT = 6  # 또는 10
    
    analyze_keypoint_trajectory(INPUT_DIR, TARGET_KEYPOINT, OUTPUT_DIR)