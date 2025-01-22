import cv2
from typing import Tuple
import numpy as np
from config.models import MODELS

class Preprocessing:
    """
    A class used to handle image preprocessing for YOLOv8n model.
    Attributes
    ----------
    image_path : str
        The file path to the image to be processed.
    Methods
    -------
    load_image() -> 'numpy.ndarray':
        Loads the image from the given file path.
    preprocess_image(img: 'numpy.ndarray') -> Tuple['numpy.ndarray', float, float]:
        Preprocesses the image for YOLOv8n model, including resizing, normalizing, and changing data layout.
    """
    def __init__(self, image_path: str):
        """
        Parameters
        ----------
        image_path : str
            The file path to the image to be processed.
        yolov8n_size : int
            The size to which the image will be resized for YOLOv8n model.
        """
        self.image_path = image_path
        self.yolov8n_size = MODELS["yoloface_8n"]["size"][0]

    def load_image(self) -> np.ndarray:
        """
        Loads the image from the given file path.
        Returns
        -------
        numpy.ndarray
            The loaded image.
        """
        img = cv2.imread(self.image_path)
        return img
    
    def preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Preprocesses the image for YOLOv8n model, including resizing, normalizing, and changing data layout.
        Parameters
        ----------
        img : numpy.ndarray
            The image to be preprocessed.
        Returns
        -------
        Tuple[numpy.ndarray, float, float]
            The preprocessed image, the width ratio, and the height ratio.
        """
        # Refer to https://github.com/TMASynthetics/facefusion/blob/main/facefusion/face_detector.py#L231
        # To refine the preprocessing
        height, width, _ = img.shape
        ratio_width = width / self.yolov8n_size
        ratio_height = height / self.yolov8n_size

        input_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = input_data.transpose(2, 0, 1)  # Change data layout from HWC to CHW
        input_data = input_data.astype('float32')
        input_data = input_data / 255.0  # Normalize to [0, 1]
        input_data = cv2.resize(input_data.transpose(1, 2, 0), (self.yolov8n_size, self.yolov8n_size)).transpose(2, 0, 1)
        
        return input_data, ratio_width, ratio_height