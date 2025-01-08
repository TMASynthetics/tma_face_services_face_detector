from src.pipeline import Pipeline

# Constants
IMAGE_PATH = "woboy.jpeg"
MODEL_PATH = ".assets/models/yoloface_8n.onnx"
YOLOV8N_SIZE = 640
DETECTION_THRESHOLD = 0.75

# Create and run the pipeline
pipeline = Pipeline(IMAGE_PATH, MODEL_PATH, YOLOV8N_SIZE, DETECTION_THRESHOLD)
results = pipeline.run()

# Print or use the results
print(results)