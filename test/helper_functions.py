import cv2

# Function to draw boxes, scores, and landmarks
def draw_results(image_path, results):
    image = cv2.imread(image_path)
    for box, score, landmarks in zip(results["bounding_boxs"], results["scores"], results["landmarks"]):
        print(box[0], box[1], box[2], box[3])
        # Draw the bounding box
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # Draw the score
        cv2.putText(image, f"{score:.2f}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw the landmarks
        for (x, y) in landmarks:
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

    return image

def save_image(image, output_path):
    # Save the annotated image
    cv2.imwrite(output_path, image)