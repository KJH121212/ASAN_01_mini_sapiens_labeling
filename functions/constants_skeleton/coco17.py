# ===============================================================
# coco17.py - COCO 17 keypoints 구조 (Sapiens/COCO 호환 버전)
# ===============================================================

# ---------------------------------------------------------------
# 🎨 색상 설정
# ---------------------------------------------------------------
COLOR_SK       = (50, 50, 50)    # Skeleton line (dark gray)
COLOR_L        = (255, 0, 0)     # Left side keypoints (red)
COLOR_R        = (0, 0, 255)     # Right side keypoints (blue)
COLOR_NEUTRAL  = (0, 255, 0)     # Central keypoints (green)

# ---------------------------------------------------------------
# 🦴 Keypoint 인덱스 (COCO 공식 17개)
# ---------------------------------------------------------------
# 0: nose, 1–4: eyes/ears, 5–16: body keypoints
LEFT_POINTS    = [5, 7, 9, 11, 13, 15]   # left shoulder–elbow–wrist–hip–knee–ankle
RIGHT_POINTS   = [6, 8, 10, 12, 14, 16]  # right shoulder–elbow–wrist–hip–knee–ankle
EXCLUDE_POINTS = [0, 1, 2, 3, 4]   # ✅ 얼굴 keypoints (nose, eyes, ears) 제외

# ---------------------------------------------------------------
# 🔗 Skeleton 연결 관계 (COCO 공식 17점 구조)
# ---------------------------------------------------------------
SKELETON_LINKS = [
    # 하체
    (15, 13), (13, 11),     # Left leg
    (16, 14), (14, 12),     # Right leg
    (11, 12),               # Hip connection

    # 상체
    (5, 11), (6, 12),       # Hip–Shoulder 연결
    (5, 6),                 # Shoulders 연결

    # 팔
    (5, 7), (7, 9),         # Left Arm
    (6, 8), (8, 10),        # Right Arm
]

# ---------------------------------------------------------------
# 📍 Keypoint 이름 (COCO 순서)
# ---------------------------------------------------------------
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

# ---------------------------------------------------------------
# 🧭 YOLO12 매핑 관계 (COCO index → YOLO12 index)
# ---------------------------------------------------------------
# YOLO12 모델로 변환 시 사용 (머리 제외)
COCO_TO_YOLO12 = {
    5: 0, 6: 1,   # shoulders
    7: 2, 8: 3,   # elbows
    9: 4, 10: 5,  # wrists
    11: 6, 12: 7, # hips
    13: 8, 14: 9, # knees
    15: 10, 16: 11  # ankles
}
