import onnxruntime as ort
import numpy as np
from config.models import MODELS

class Inference:
    """
    A class used to perform inference using a pre-trained model.

    Attributes
    ----------
    model_path : str
        The path to the pre-trained model file.
    session : ort.InferenceSession
        The ONNX Runtime inference session initialized with the model.

    Methods
    -------
    run_inference(input_data: np.ndarray) -> np.ndarray
        Runs inference on the provided input data and returns the model outputs.
    """

    def __init__(self):
        self.model_path = MODELS["yoloface_8n"]["path"]
        self.session = ort.InferenceSession(self.model_path)

    def run_inference(self, input_data: np.ndarray) -> np.ndarray:
        """
        Runs inference on the provided input data and returns the model outputs.

        Parameters
        ----------
        input_data : np.ndarray
            The input data to run inference on.

        Returns
        -------
        np.ndarray
            The outputs from the model inference.
        """
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_data[None, :]})
        return outputs