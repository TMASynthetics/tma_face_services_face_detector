from fastapi import FastAPI
from fastapi import File, UploadFile
import cv2
import io
from facefusion.face_detector import detect_faces
import numpy as np

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Process the image here
        bounding_boxes, face_scores, face_landmarks_5 = detect_faces(image)
        return {
            "filename": file.filename, 
            "format": image.format, 
            "mode": image.mode, 
            "size": image.size,
            "bounding_boxes": bounding_boxes,
            "face_scores": face_scores,
            "face_landmarks_5": face_landmarks_5
        }
    except (IOError, SyntaxError) as e:
        return {"error": f"{e}\n The uploaded file is most likely not a valid image."}