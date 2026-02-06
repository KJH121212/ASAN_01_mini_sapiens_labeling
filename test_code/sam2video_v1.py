import cv2
import json
import glob
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def rle_to_mask(rle, height, width):
    """
    [start, length, start, length...] 형태의 RLE 리스트를 마스크로 변환
    """
    mask = np.zeros(height * width, dtype=np.uint8)
    if not rle: 
        return mask.reshape((height, width))
    
    rle = np.array(rle)
    starts = rle[0::2]
    lengths = rle[1::2]
    starts = starts - 1 
    ends = starts + lengths
    
    for lo, hi in zip(starts, ends):
        if lo < 0: lo = 0
        if hi > len(mask): hi = len(mask)
        mask[lo:hi] = 1
        
    return mask.reshape((height, width))

def render_segmentation_video(frame_dir: str, json_dir: str, output_path: str, fps: int = 30, alpha: float = 0.5):
    """
    프레임 이미지와 JSON 세그멘테이션 데이터를 결합하여 비디오를 생성합니다.
    """
    frame_path_obj = Path(frame_dir)
    json_path_obj = Path(json_dir)
    frame_files = sorted(glob.glob(str(frame_path_obj / "*.jpg")))
    
    if not frame_files:
        print(f"❌ 에러: {frame_dir}에서 이미지를 찾을 수 없습니다.")
        return

    first_img = cv2.imread(frame_files[0])
    height, width = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    color_map = {}
    def get_color(obj_id):
        if obj_id not in color_map:
            color_map[obj_id] = [random.randint(100, 255) for _ in range(3)]
        return color_map[obj_id]

    print(f"⏳ 렌더링 시작: {output_path}")
    for i, img_path in enumerate(tqdm(frame_files, desc="Rendering")):
        frame = cv2.imread(img_path)
        if frame is None: continue

        overlay = frame.copy()
        json_filename = f"{i:06d}.json"
        json_file = json_path_obj / json_filename
        
        bbox_draw_infos = []
        shapes_found = False

        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if "objects" in data:
                    for obj in data["objects"]:
                        shapes_found = True
                        obj_id = obj.get("id", 0)
                        rle_counts = obj["segmentation"]["counts"]
                        
                        mask = rle_to_mask(rle_counts, height, width)
                        
                        if mask.sum() > 0:
                            color = get_color(obj_id)
                            
                            # 1) Segmentation 마스크 그리기
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.fillPoly(overlay, contours, color)

                            # 2) BBox 계산을 위해 모든 컨투어 합치기 [수정된 부분]
                            if contours:
                                # [수정] 제너레이터 대신 튜플(contours) 자체를 넘기거나 리스트로 변환해야 함
                                # contours는 이미 튜플이므로 np.vstack(contours)로 충분합니다.
                                all_contours = np.vstack(contours)
                                x, y, w, h = cv2.boundingRect(all_contours)
                                
                                bbox_draw_infos.append({
                                    'bbox': (x, y, w, h),
                                    'color': color,
                                    'id': obj_id
                                })

            except Exception as e:
                # 에러 메시지를 더 자세히 출력
                print(f"⚠️ 프레임 {i} JSON 오류: {e}")

        # 5. 오버레이 합성
        if shapes_found:
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 6. BBox 및 ID 라벨 그리기
        for info in bbox_draw_infos:
            x, y, w, h = info['bbox']
            color = info['color']
            obj_id = info['id']
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            label = f"ID {obj_id}"
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            text_x = x
            text_y = y - 5 
            if text_y - text_height < 0:
                text_y = y + h + text_height + 10

            cv2.rectangle(frame, 
                          (text_x, text_y - text_height - baseline), 
                          (text_x + text_width, text_y + baseline), 
                          color, -1)
            
            cv2.putText(frame, label, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # 7. 카운터 표시
        counter_text = f"Frame: {i}/{len(frame_files)}"
        cv2.putText(frame, counter_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(frame, counter_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

        out.write(frame)

    out.release()
    print(f"🎉 렌더링 완료: {output_path}")

# 실행 부는 동일하게 유지
if __name__ == "__main__":
    target = 1011

    DADA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
    CSV_PATH = DADA_DIR / "metadata.csv"
    df = pd.read_csv(CSV_PATH)
    COMMON_PATH = df.loc[target, "common_path"]
    
    FRAME_DIR = DADA_DIR / "1_FRAME" / COMMON_PATH
    JSON_DIR = DADA_DIR / "8_SAM" / COMMON_PATH
    OUTPUT_VIDEO_PATH = DADA_DIR / f"test/{COMMON_PATH}/SAM_results_bbox.mp4"
    OUTPUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"▶️ 처리 대상: {COMMON_PATH}")

    render_segmentation_video(
        frame_dir=str(FRAME_DIR),
        json_dir=str(JSON_DIR),
        output_path=str(OUTPUT_VIDEO_PATH),
        fps=30,
        alpha=0.5
    )