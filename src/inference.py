import onnxruntime as ort

class Inference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = ort.InferenceSession(self.model_path)

    def run_inference(self, input_data):
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_data[None, :]})
        return outputs