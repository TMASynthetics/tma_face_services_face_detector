Command for triton server download
```bash
podman pull nvcr.io/nvidia/tritonserver:24.12-py3
```

Your directory with all your onnx models must follow this architecture:
models/
└── mon_modele/
    ├── 1/
    │   └── model.onnx
    └── config.pbtxt

