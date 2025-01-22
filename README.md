# Face Detector Service
This project implements a face detection pipeline using a YOLO-based model. The pipeline includes preprocessing, inference, and postprocessing steps to detect faces in images.

# Project Structure
```
/home/quillaur/bethel/tma_face_services_face_detector/
├── README.md
├── requirements.txt
├── config/
│   ├── models.py
├── src/
│   ├── inference.py
│   ├── pipeline.py
│   ├── posprocessing.py
│   └── preprocessing.py
└── test/
    ├── test_data/
    |   ├── request.json
    |   └── marie.jpeg
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

# Load the request data from the JSON file
with open('test/test_data/request.json', 'r') as request_file:
	request_data = json.load(request_file)

# Create and run the pipeline
pipeline = Pipeline(request_data["args"]["input_image_1"])
results = pipeline.run()

# Write the output data to a JSON file
with open(request_data["args"]["output_image_data"], 'w') as outfile:
	json.dump(results, outfile)

# Print or use the results
print("Number of faces:", len(results["bounding_boxes"]))
print("Bounding box:", results["bounding_boxes"])
print("Score:", results["scores"])
print("Landmarks:", results["landmarks"]["5"])
```

See "test/test_marie.py" for a similar running exemple with drawing the results on the image.

# Inside the pipeline
Granted that you have the input, here is what is going on step by step when you call pipeline.run():

   ```mermaid
   graph TD
      subgraph Inputs
      RJ[Request JSON] --> II[Input Image]
      end

      subgraph Preprocessing
      II --> Normalize
      Normalize --> Resize
      Resize --> PI[Preprocessed Image] & AR[Aspect Ratio]
      end
      
      subgraph Inference
      PI --> RFDI(Run Face Detector)
      RFDI --> RBB[Raw Bounding Boxes] & RS[Raw Scores] & RFL5[Raw Face Landmarks 5]
      end

      subgraph Postprocessing
      RBB & RS & RFL5 --> FDT[Filter by Detection Threshold]
      FDT & AR --> ARA[Aspect Ratio Adjustement] 
      ARA --> FOF[Filter Overlapping Faces]
      end

      subgraph Outputs
      FOF --> BB[Bounding Boxes] & S[Scores] & L5[Landmarks 5]
      end
   ```

