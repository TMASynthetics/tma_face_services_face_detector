from src.preprocessing import Preprocessing
from src.inference import Inference
from src.postprocessing import Postprocessing
from typing import Any, List, Tuple

class Pipeline:
    """
    A class to represent a pipeline for face detection using YOLOv8.

    Attributes:
    -----------
    image_path : str
        Path to the input image.
    model_path : str
        Path to the YOLOv8 model.
    yolov8n_size : Tuple[int, int]
        Size of the YOLOv8 input.
    detection_threshold : float
        Threshold for detection confidence.

    Methods:
    --------
    run() -> List[Any]:
        Executes the pipeline: preprocessing, inference, and postprocessing.
    """
    def __init__(self, image_path: str, model_path: str, yolov8n_size: Tuple[int, int], detection_threshold: float):
        """
        Constructs all the necessary attributes for the Pipeline object.

        Parameters:
        -----------
        image_path : str
            Path to the input image.
        model_path : str
            Path to the YOLOv8 model.
        yolov8n_size : Tuple[int, int]
            Size of the YOLOv8 input.
        detection_threshold : float
            Threshold for detection confidence.
        """
        self.image_path = image_path
        self.model_path = model_path
        self.yolov8n_size = yolov8n_size
        self.detection_threshold = detection_threshold

    def run(self) -> List[Any]:
        """
        Executes the pipeline: preprocessing, inference, and postprocessing.

        Returns:
        --------
        List[Any]
            The results of the face detection.
        """
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