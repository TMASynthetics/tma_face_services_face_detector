# tma_face_services_face_detector
Isolate face detector processor into one microservice.

# Installation
```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

# Run
```python
from src.pipeline import Pipeline

# Constants
IMAGE_PATH = "your_image.jpeg"
MODEL_PATH = "<where your model is>/yoloface_8n.onnx"
YOLOV8N_SIZE = 640
DETECTION_THRESHOLD = 0.75

# Create and run the pipeline
pipeline = Pipeline(IMAGE_PATH, MODEL_PATH, YOLOV8N_SIZE, DETECTION_THRESHOLD)
results = pipeline.run()

# Print or use the results
print(results)
```