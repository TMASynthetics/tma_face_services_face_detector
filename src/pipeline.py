from src.preprocessing import Preprocessing
from src.inference import Inference
from src.postprocessing import Postprocessing
from typing import List, Tuple
import numpy as np

class Pipeline:
    """
    A class to represent a pipeline for face detection using YOLOv8.

    Attributes:
    -----------
    image_path : str
        Path to the input image.

    Methods:
    --------
    run() -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
        Executes the pipeline: preprocessing, inference, and postprocessing.
    """
    def __init__(self, image_path: str):
        """
        Constructs all the necessary attributes for the Pipeline object.

        Parameters:
        -----------
        image_path : str
            Path to the input image.
        """
        self.image_path = image_path

    def run(self) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
        """
        Executes the pipeline: preprocessing, inference, and postprocessing.

        Returns:
        --------
        Tuple[List[np.ndarray], List[float], List[np.ndarray]]
            The results of the face detection.
        """
        # Preprocessing
        preprocessor = Preprocessing(self.image_path)
        img = preprocessor.load_image()
        input_data, ratio_width, ratio_height = preprocessor.preprocess_image(img)

        # Inference
        inference = Inference()
        outputs = inference.run_inference(input_data)

        # Postprocessing
        postprocessor = Postprocessing()
        results = postprocessor.process_outputs(outputs, ratio_width, ratio_height)

        # For JSON serialisation purpose, we need to convert numpy arrays to lists
        # Transform any numpy array to list in the self.response dict
        def transform_numpy_to_list(obj):
            if isinstance(obj, dict):
                return {k: transform_numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [transform_numpy_to_list(i) for i in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
		
        results = transform_numpy_to_list(results)

        # Return or print the results
        return results