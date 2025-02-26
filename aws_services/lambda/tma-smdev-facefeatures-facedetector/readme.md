This directory aims to simulate an AWS environment to ensure the functionality of our pipeline.

- `lambda_function.py`: Contains the core logic of the program within a Lambda function.
- `model_api.py`: Simulates SageMaker calls by serving model inference using FastAPI and ONNX Runtime.
- `moto_test.py`: Simulates the invocation of the Lambda function with a mock S3 bucket.

