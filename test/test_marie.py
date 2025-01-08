import sys
import os

# Add the src directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import Pipeline
from helper_functions import draw_results, save_image

# Constants
IMAGE_PATH = "test/test_data/marie.jpeg"
MODEL_PATH = ".assets/models/yoloface_8n.onnx"
YOLOV8N_SIZE = 640
DETECTION_THRESHOLD = 0.75

# Create and run the pipeline
pipeline = Pipeline(IMAGE_PATH, MODEL_PATH, YOLOV8N_SIZE, DETECTION_THRESHOLD)
results = pipeline.run()

# Print or use the results
print(results)

output_path = "test/test_data/marie_annotated.jpeg"
# Draw the results on the image
annotated_image = draw_results(IMAGE_PATH, results)
save_image(annotated_image, output_path)