MODELS_DIR_PATH = ".../<path_to_model_dir>"

MODELS = {
    "yoloface_8n":
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yoloface_8n.onnx',
        'path': f'{MODELS_DIR_PATH}/yoloface_8n.onnx',
        'size': (640, 640),
        "detection_threshold": 0.75,
    }
}
