# ===============================================================
# registry.py - skeleton 설정 자동 로딩
# ===============================================================
from importlib import import_module

AVAILABLE_MODELS = {
    "coco17": "functions.constants_skeleton.coco17",
    "yolo12": "functions.constants_skeleton.yolo12",
}

def load_skeleton_constants(model_type: str):
    """모델 타입에 따라 skeleton constants 동적 로딩"""
    if model_type not in AVAILABLE_MODELS:
        raise ValueError(f"❌ Unknown skeleton model type: {model_type}")
    return import_module(AVAILABLE_MODELS[model_type])
