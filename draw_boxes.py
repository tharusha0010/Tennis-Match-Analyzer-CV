import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="5tedwy8cTM2JfRId8RHu"
)

image_path = r"D:\Projects\Tennis-Match-Analyzer-CV\extracted_frames\frame_0046.jpg"
image = cv2.imread(image_path)

result = CLIENT.infer(image_path, model_id="tennis-match-analyzer-cv-bbwo3/1")

detections = sv.Detections.from_inference(result)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

cv2.imshow("Tennis Match Analysis", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()