import os, sys, time, json

# Add the src directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import Pipeline
from helper_functions import draw_results, save_image

# Record the start time of the execution
start_time = time.time()

# Load the request data from the JSON file
with open('test/test_data/request.json', 'r') as request_file:
	request_data = json.load(request_file)

# Create and run the pipeline
pipeline = Pipeline(request_data["args"]["input_image_1"])
results = pipeline.run()

# Record the end time of the execution
end_time = time.time()

# Print the execution time
print(f"Execution time: {round(end_time - start_time, 2)} seconds")

# Write the output data to a JSON file
with open(request_data["args"]["output_image_data"], 'w') as outfile:
	json.dump(results, outfile)

# Print or use the results
print("Number of faces:", len(results["bounding_boxes"]))
print("Bounding box:", results["bounding_boxs"])
print("Score:", results["scores"])
print("Landmarks:", results["landmarks"])

output_path = "test/test_data/marie_annotated.jpeg"
# Draw the results on the image
annotated_image = draw_results(request_data["args"]["input_image_1"], results)
save_image(annotated_image, output_path)