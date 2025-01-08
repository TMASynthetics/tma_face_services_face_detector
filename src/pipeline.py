from src.preprocessing import Preprocessing
from src.inference import Inference
from src.postprocessing import Postprocessing

class Pipeline:
    def __init__(self, image_path, model_path, yolov8n_size, detection_threshold):
        self.image_path = image_path
        self.model_path = model_path
        self.yolov8n_size = yolov8n_size
        self.detection_threshold = detection_threshold

    def run(self):
        # Preprocessing
        preprocessor = Preprocessing(self.image_path, self.yolov8n_size)
        img = preprocessor.load_image()
        input_data, ratio_width, ratio_height = preprocessor.preprocess_image(img)

        # Inference
        inference = Inference(self.model_path)
        outputs = inference.run_inference(input_data)

        # Postprocessing
        postprocessor = Postprocessing(self.detection_threshold)
        results = postprocessor.process_outputs(outputs, ratio_width, ratio_height)

        # Return or print the results
        return results