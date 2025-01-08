import cv2

class Preprocessing:
    def __init__(self, image_path, yolov8n_size):
        self.image_path = image_path
        self.yolov8n_size = yolov8n_size

    def load_image(self):
        img = cv2.imread(self.image_path)
        return img

    def preprocess_image(self, img):
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