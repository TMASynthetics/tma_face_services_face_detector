# Face Detector Service
This project implements a face detection pipeline using a YOLO-based model. The pipeline includes preprocessing, inference, and postprocessing steps to detect faces in images.

# Project Structure
```
/home/quillaur/bethel/tma_face_services_face_detector/
├── README.md
├── requirements.txt
├── src/
│   ├── inference.py
│   ├── pipeline.py
│   ├── posprocessing.py
│   └── preprocessing.py
└── test/
    ├── test_data/
    ├── helper_functions.py
    └── test_marie.py
```

# Installation
```bash
git clone <repository_url>
cd <repository_directory>
```
```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```
If need be, you can download the `yoloface_8n.onnx` model [here](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yoloface_8n.onnx).

Ensure you have the model file `yoloface_8n.onnx` saved somewhere in this project.

# Run
You can run the face detection pipeline using the `Pipeline` class. 

Here is an example:

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
for result in results:
    print("Bounding box:", result[0])
    print("Score:", result[1])
    print("Landmarks:", result[2])
```

See "test/test_marie.py" for a similar running exemple with drawing the results on the image.

