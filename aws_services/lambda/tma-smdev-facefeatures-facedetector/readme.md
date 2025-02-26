WARNING ! => Triton server is very big if downloaded by default !
https://github.com/triton-inference-server/server
https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/compose.md

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

sudo podman --root /home/quillaur/HDD-1TO/containers/storage --runroot /home/quillaur/HDD-1TO/containers/runroot pull nvcr.io/nvidia/tritonserver:25.01-py3

sudo podman --root /home/quillaur/HDD-1TO/containers/storage --runroot /home/quillaur/HDD-1TO/containers/runroot run --device nvidia.com/gpu=all -it --rm nvcr.io/nvidia/tritonserver:25.01-py3
sudo podman --root /home/quillaur/HDD-1TO/containers/storage --runroot /home/quillaur/HDD-1TO/containers/runroot run -it --rm nvcr.io/nvidia/tritonserver:25.01-py3

sudo podman --root /home/quillaur/HDD-1TO/containers run \
  --rm -p 8001:8001 \
  --gpus all \
  -v /home/quillaur/HDD-1TO/models/bethel:/models \
  nvcr.io/nvidia/tritonserver:25.01-py3 \
  tritonserver --model-repository=/models