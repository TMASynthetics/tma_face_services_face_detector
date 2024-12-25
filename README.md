# tma_face_services_face_detector
Isolate face detector processor into one microservice.

# Installation
```bash
conda init --all
conda env create -f environment.yml
conda activate face_detector
```

# Run
```bash
fastapi dev main.py
```
It runs locally on: http://127.0.0.1:8000 