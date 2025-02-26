import onnxruntime as ort
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Load ONNX model
model_path = ".../yoloface_8n.onnx"  # Update with your actual model path
try:
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
except Exception as e:
    raise RuntimeError(f"Failed to load ONNX model: {e}")

class InferenceRequest(BaseModel):
    data: list

@app.post("/predict")
async def predict(request: InferenceRequest):
    try:
        # Convert input data to numpy array
        input_data = np.array(request.data, dtype=np.float32)

        # Run inference
        output = session.run(None, {input_name: input_data[None, :]})
        return {"predictions": output[0].tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
