import os
import json

def extract_bbox_and_id(json_file_path):
    """
    JSON 파일 하나를 읽어서 (file_name, 객체 리스트[{id, bbox}])를 반환하는 함수
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            
        file_name = data.get('file_name', 'Unknown_File')
        objects = data.get('objects', [])
        
        extracted_results = []
        
        for obj in objects:
            # ID 추출 (없을 경우 'No-ID'로 표시)
            obj_id = obj.get('id', 'No-ID')
            
            # BBox 추출 (없을 경우 빈 리스트)
            bbox = obj.get('bbox', [])
            
            # 결과 리스트에 추가
            extracted_results.append({
                'id': obj_id,
                'bbox': bbox
            })
            
        return file_name, extracted_results

    except Exception as e:
        print(f"❌ 읽기 에러 ({os.path.basename(json_file_path)}): {e}")
        return None, []
