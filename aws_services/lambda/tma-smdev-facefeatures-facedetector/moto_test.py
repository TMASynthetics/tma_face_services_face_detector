import logging, json
import boto3
from moto import mock_aws
from lambda_function import lambda_handler

# =============================
# LOGGING CONFIGURATION
# =============================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -------------------------------
# Test Using moto Mocks
@mock_aws
def test_lambda_handler():
    try:
        logger.info("==== Lambda test initiated ====")
        # Create a mock S3 bucket and upload a test image
        s3 = boto3.client('s3')
        bucket_name = 'test-bucket'
        s3.create_bucket(Bucket=bucket_name)
        
        # Upload the image to the mock S3 bucket
        # to act as if it is already there.
        with open('test/test_data/marie.jpeg', 'rb') as image_file:
            s3.put_object(Bucket=bucket_name, Key='source.jpg', Body=image_file.read())
        
        # Create a test event
        event = {
            "bucketName": bucket_name,
            "sourceKey": "source.jpg"
        }
        
        # Call the lambda_handler function
        result = lambda_handler(event, None)
        
        # Check the result
        assert result['statusCode'] == 200
        body = json.loads(result['body'])
        assert 'output_key' in body
        assert body['message'] == "Face detection done"
    
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
    finally:
        logger.info("===== Test Lambda function finished =====")


if __name__ == "__main__":
    # Run the test
    test_lambda_handler()
