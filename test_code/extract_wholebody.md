# version 별 특징
## V4.0
1. 입력 : [원본 이미지] + [SAM BBox]  
      ⬇️  
2. 전처리:(1.2배 확대 -> Crop -> 1024x768 Resize -> Normalize)  
      ⬇️  
3. Pose estimate: [Sapiens 모델 (GPU)] -> Heatmap 출력  
      ⬇️  
3. 후처리: (Argmax로 좌표 추출 -> Resize 역변환 -> Crop 위치 보정)  
      ⬇️  
4. 저장: [원본 좌표 Keypoints] -> JSON 저장  
### V4.1
최종 저장 json의 bbox를 후처리 이후 bbox가 아닌 전처리 후 bbox로 변경
### v4.2
2->3 제공하는 crop 이미지 bbox 크기 변경  
변경 전 : 각 frame 마다의 bbox * 1.2  
변경 후 : 모든 frame의 bbox를 포함하는 가장 큰 bbox*1.2 로 모든 프레임 고정.
-> 문제점 발생: 환자가 이동하는 경우
### v4.3
2->3 제공하는 crop 이미지 bbox 크기 변경  
변경 전 : 각 frame 마다의 bbox * 1.2  
변경 후 : 첫 프레임 BBox의 중심을 유지하면서, 긴 변의 1.55배 크기를 가진 정사각형 고정  
-> 문제점 발생: 환자가 이동하는 경우
### v4.4
crop 후 resize 가 아니라 padding을 통해 회색 영역을 채워 넣어 이미지의 비율을 유지.  
argmax 방식으로 heatmap에서 가장 값이 큰 좌표를 찾는데 결과rk 항상 정수 픽셀로 나오게 됨.  
소수점 단위까지 위치를 보정하도록 변경  
### v5.0
Full Instalation 방식으로 변경  
4.1과 같이 bbox crop 까지는 동일하나, resize가 아닌 회색으로 padding하는 방식을 통해 비율을 고정시킴.  

### v5.1
기존 코드
MMDetection 기반의 Standard Detectior 사용  
이미지의 미세한 픽셀 노이즈나 조명 변화에 덜 민감하여 Bbox가 덩어리로서 안정적으로 유지됨  
Bbox 좌표의 변화가 부드러움

신형 코드
SAM 마스트에서 Bbox 추출  
픽셀 단위 Segmentation을 수행하기 때문에 edge가 매 프레임 미세하게 변함.  
Mask2bbox 과정에서 1개의 픽셀만 튀어도 즉각적으로 bbox가 변함. 

개선 방향
기본적으로 첫 frame * 1.2의 bbox로 고정.  
- 조건 1: 마스크(객체)가 box를 나가는 즉시 box의 크기 조정
- 조건 2: 마스크가 해당 공간에서 차지하는 70% 이하면 bbox 크기 줄이기.
