import numpy as np

class Postprocessing:
    def __init__(self, detection_threshold):
        self.detection_threshold = detection_threshold

    def process_outputs(self, outputs, ratio_width, ratio_height):
        # Taken from https://github.com/TMASynthetics/facefusion/blob/main/facefusion/face_detector.py#L231
        detection = np.squeeze(outputs).T
        bounding_box_raw, score_raw, face_landmark_5_raw = np.split(detection, [ 4, 5 ], axis = 1)

        keep_indices = np.where(score_raw > self.detection_threshold)[0]

        bounding_boxes = []
        face_scores = []
        face_landmarks_5 = []

        if np.any(keep_indices):
            bounding_box_raw, face_landmark_5_raw, score_raw = bounding_box_raw[keep_indices], face_landmark_5_raw[keep_indices], score_raw[keep_indices]

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

        return bounding_boxes, face_scores, face_landmarks_5