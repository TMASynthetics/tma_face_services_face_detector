import numpy as np
from typing import List, Tuple
from config.models import MODELS


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
                "bounding_boxs": non_overlapping_boxes, 
                "scores": non_overlapping_scores, 
                "landmarks": non_overlapping_landmarks
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
    