import cv2
import supervision as sv
import os
from inference import get_model

MODEL_ID = "tennis-match-analyzer-cv/2"
API_KEY = "rWJd7YhKrAjVu4Z395W7"

model = get_model(model_id=MODEL_ID, api_key=API_KEY)

# 2. Setup paths
SOURCE_VIDEO_PATH = r"D:\Projects\Tennis-Match-Analyzer-CV\input_videos\test_video.mp4"
OUTPUT_DIR = r"D:\Projects\Tennis-Match-Analyzer-CV\output_videos"
TARGET_VIDEO_PATH = os.path.join(OUTPUT_DIR, "output_test_video.mp4")

os.makedirs(OUTPUT_DIR, exist_ok=True)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def process_frame(frame, _):
    result = model.infer(frame)[0]
    
    detections = sv.Detections.from_inference(result)
    
    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
    
    return annotated_frame

print("Local video analysis starting... please wait.")

sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=process_frame
)

print(f"\nSuccess! View the new video here: {TARGET_VIDEO_PATH}")