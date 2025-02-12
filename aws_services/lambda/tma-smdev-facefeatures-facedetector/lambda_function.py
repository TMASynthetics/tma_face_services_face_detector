import logging
import boto3
import numpy as np
import time
from typing import Any, Dict, List, Tuple
from numpy.typing import NDArray
import cv2
import json
import onnxruntime as ort

# =============================
# LOGGING CONFIGURATION
# =============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =============================
# BOTO3 GLOBAL SESSION & CLIENTS
# =============================
# Create a global Boto3 session
# global_session = boto3.Session()

# # Advanced configuration for Boto3 clients
# config = botocore.config.Config(
#     max_pool_connections=50,  # Maximum number of connections to keep in the connection pool
#     tcp_keepalive=True,       # Enable TCP keep-alive
#     connect_timeout=5,        # Timeout in seconds for establishing a connection
#     read_timeout=60,          # Timeout in seconds for reading from a connection
#     retries={'max_attempts': 0}  # Disable automatic retries
# )

# # Create an S3 client using the global session and the advanced configuration
# s3_client = global_session.client("s3", config=config)

# # Create a SageMaker Runtime client using the global session and the advanced configuration
# sagemaker_runtime = global_session.client("sagemaker-runtime", config=config)

# =============================
# TYPES & CONSTANTS
# =============================
VisionFrame = NDArray[Any]

MODELS_PATH = "/home/quillaur/HDD-1TO/models/bethel"

MODELS = {
    "yoloface_8n":
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yoloface_8n.onnx',
        'path': f'{MODELS_PATH}/yoloface_8n.onnx',
        'size': (640, 640),
        "detection_threshold": 0.75,
    }
}


# =============================
# PRE-/POST-PROCESSING CLASSES
# =============================

class Preprocessing:
    """
    A class used to handle image preprocessing for YOLOv8n model.

    Methods
    -------
    preprocess_image(img: 'numpy.ndarray') -> Tuple['numpy.ndarray', float, float]:
        Preprocesses the image for YOLOv8n model, including resizing, normalizing, and changing data layout.
    """
    def __init__(self):
        """
        Parameters
        ----------
        yolov8n_size : int
            The size to which the image will be resized for YOLOv8n model.
        """
        self.yolov8n_size = MODELS["yoloface_8n"]["size"][0]
    
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


class Postprocessing:
    def __init__(self):
        """
        Initialize the Postprocessing class with a detection threshold.
        """
        self.detection_threshold = MODELS["yoloface_8n"]["detection_threshold"]

    def process_outputs(self, outputs: np.ndarray, ratio_width: float, ratio_height: float) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
        """
        Process the outputs from the face detector model, filter detections based on the detection threshold,
        and adjust bounding boxes and landmarks according to the given width and height ratios.
        Args:
            outputs (np.ndarray): The raw outputs from the face detector model.
            ratio_width (float): The ratio to adjust the width of the bounding boxes.
            ratio_height (float): The ratio to adjust the height of the bounding boxes.
        Returns:
            Tuple[List[np.ndarray], List[float], List[np.ndarray]]: A tuple containing the non-overlapping bounding boxes,
            their corresponding scores, and the landmarks for each face.
        """
        # Taken from https://github.com/TMASynthetics/facefusion/blob/main/facefusion/face_detector.py#L231
        detection = np.squeeze(outputs).T
        bounding_box_raw, score_raw, face_landmark_5_raw = np.split(detection, [ 4, 5 ], axis = 1)

        keep_indices = np.where(score_raw > self.detection_threshold)[0]

        bounding_boxes = []
        face_scores = []
        face_landmarks_5 = []

        if np.any(keep_indices):
            bounding_box_raw, face_landmark_5_raw, score_raw = bounding_box_raw[keep_indices], face_landmark_5_raw[keep_indices], score_raw[keep_indices]

            # Adjust the bounding boxes according to the width and height ratios
            for bounding_box in bounding_box_raw:
                bounding_boxes.append(np.array(
                [
                    (bounding_box[0] - bounding_box[2] / 2) * ratio_width,
                    (bounding_box[1] - bounding_box[3] / 2) * ratio_height,
                    (bounding_box[0] + bounding_box[2] / 2) * ratio_width,
                    (bounding_box[1] + bounding_box[3] / 2) * ratio_height,
                ]))

            face_scores = score_raw.ravel().tolist()
            face_landmark_5_raw[:, 0::3] = (face_landmark_5_raw[:, 0::3]) * ratio_width
            face_landmark_5_raw[:, 1::3] = (face_landmark_5_raw[:, 1::3]) * ratio_height

            for face_landmark_5 in face_landmark_5_raw:
                face_landmarks_5.append(np.array(face_landmark_5.reshape(-1, 3)[:, :2]))
        
        indices_to_keep = self.filter_overlapping_boxes(bounding_boxes, face_scores)
        non_overlapping_boxes = [bounding_boxes[i] for i in indices_to_keep]
        non_overlapping_scores = [face_scores[i] for i in indices_to_keep]
        non_overlapping_landmarks = [face_landmarks_5[i] for i in indices_to_keep]

        return {
                "bounding_boxes": non_overlapping_boxes, 
                "scores": non_overlapping_scores, 
                "landmarks": {
                    "5": non_overlapping_landmarks
                }
            }
    
    def filter_overlapping_boxes(self, bounding_boxes: List[np.ndarray], face_scores: List[float], iou_threshold: float = 0.7) -> set:
        """
        Filter out overlapping bounding boxes based on Intersection over Union (IoU) threshold.
        Args:
            bounding_boxes (List[np.ndarray]): List of bounding boxes.
            face_scores (List[float]): List of scores corresponding to each bounding box.
            iou_threshold (float, optional): The IoU threshold to determine overlapping boxes. Defaults to 0.7.
        Returns:
            set: Indices of the bounding boxes to keep.
        """
        def compute_iou(box1, box2):
            x1_max = max(box1[0], box2[0])
            y1_max = max(box1[1], box2[1])
            x2_min = min(box1[2], box2[2])
            y2_min = min(box1[3], box2[3])

            inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = box1_area + box2_area - inter_area

            return inter_area / union_area

        indices_to_keep = set(range(len(bounding_boxes)))
        for i in range(len(bounding_boxes)):
            for j in range(i + 1, len(bounding_boxes)):
                if compute_iou(bounding_boxes[i], bounding_boxes[j]) > iou_threshold:
                    if face_scores[i] > face_scores[j]:
                        indices_to_keep.discard(j)
                    else:
                        indices_to_keep.discard(i)

        return indices_to_keep
  

class Pipeline:

    def run(self, source_image: VisionFrame) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
        # Preprocessing
        preprocessor = Preprocessing()
        input_data, ratio_width, ratio_height = preprocessor.preprocess_image(source_image)

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


# =============================
# LAMBDA HANDLER
# =============================
def lambda_handler(event, context):
    overall_start = time.time()
    logger.info("===== Lambda execution started =====")
    logger.info(f"Received event: {event}")

    try:
        # -------------------------------------------------------
        # STEP 1: Retrieve S3 information and parameters
        # -------------------------------------------------------
        s3_client = boto3.client('s3')

        bucket_name = event.get("bucketName")
        # target_key = event.get("targetKey")
        source_key = event.get("sourceKey")

        # Retrieve landmarks and embedding from the event if provided
        # source_landmark_5 = event.get("source_landmark_5", [])
        # target_landmark_5 = event.get("target_landmark_5", [])
        # source_embedding = event.get("source_embedding", [])

        # if not all([bucket_name, target_key, source_key]):
        if not all([bucket_name, source_key]):
            raise ValueError("The keys 'bucketName', 'sourceKey' are missing from the event.")

        # -------------------------------------------------------
        # STEP 2: Download images from S3
        # -------------------------------------------------------
        # local_target_path = "/tmp/target.jpg"
        local_source_path = "/tmp/source.jpg"

        t0 = time.time()
        # s3_client.download_file(bucket_name, target_key, local_target_path)
        s3_client.download_file(bucket_name, source_key, local_source_path)
        t1 = time.time()
        logger.info(f"Time spent downloading images: {t1 - t0:.4f} sec")

        # -------------------------------------------------------
        # STEP 3: Read images
        # -------------------------------------------------------
        t0 = time.time()

        try:
            source_image: VisionFrame = cv2.imread(local_source_path)
        except Exception as e:
            logger.error(f"Error reading the source image: {e}")
            raise ValueError(f"Error reading the source image: {e}")

        # if target_image is None or source_image is None:
        if source_image is None:
            logger.error("Unable to read one of the images.")
            raise ValueError("Unable to read one of the source image.")

        t1 = time.time()
        logger.info(f"Time spent reading the image: {t1 - t0:.4f} sec")

        # -------------------------------------------------------
        # STEP 4: Execute the Face Enhancement pipeline
        # -------------------------------------------------------
        t0 = time.time()
        pipeline = Pipeline()

        # Note: The method signature below is an example.
        # In your full code, you may need to adapt to pass target_image, source_image, or other parameters.
        results = pipeline.run(
            source_image=source_image
        )

        t1 = time.time()
        logger.info(f"Time spent on face enhancement pipeline: {t1 - t0:.4f} sec")

        # -------------------------------------------------------
        # STEP 5: Save or return the result
        # -------------------------------------------------------
        # We need the input (payload) and output keys to be returned.
        output = event.copy()

        # Example: Save locally and re-upload to S3
        output_path = "output.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

        output_key = event.get("outputKey", f"outputs/{output_path}")
        s3_client.upload_file(output_path, bucket_name, output_key)
        logger.info(f"Result image uploaded to s3://{bucket_name}/{output_key}")
        output["output_key"] = output_key

        output["message"] = "Face detection done"

        overall_end = time.time()
        logger.info(f"Total Lambda execution time: {overall_end - overall_start:.4f} sec")

        return {
            "statusCode": 200,
            "body": json.dumps(output)
        }

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
    finally:
        logger.info("===== Lambda execution finished =====")