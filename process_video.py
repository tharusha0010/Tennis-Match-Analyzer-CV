import os
from collections import deque
import cv2
from inference import get_model
import numpy as np
from scipy.signal import savgol_filter
import supervision as sv

MODEL_ID = "your model ID"
API_KEY = "your API key"

SOURCE_VIDEO_PATH = r"D:\Projects\Tennis-Match-Analyzer-CV\input_videos\test_video.mp4"
OUTPUT_DIR = r"D:\Projects\Tennis-Match-Analyzer-CV\output_videos"
TARGET_VIDEO_PATH = os.path.join(OUTPUT_DIR, "output_tracked_video.mp4")

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = get_model(model_id=MODEL_ID, api_key=API_KEY)
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

ball_path = deque(maxlen=15)

def interpolate_and_smooth(points):
    if len(points) < 3:
        return points

    interpolated = [points[0]]
    for i in range(1, len(points)):
        p1, p2 = points[i - 1], points[i]
        dist = np.linalg.norm(np.array(p1) - np.array(p2))

        if 20 < dist < 200:
            steps = int(dist // 10)
            for step in range(1, steps + 1):
                new_x = int(p1[0] + (p2[0] - p1[0]) * (step / (steps + 1)))
                new_y = int(p1[1] + (p2[1] - p1[1]) * (step / (steps + 1)))
                interpolated.append((new_x, new_y))
        interpolated.append(p2)

    if len(interpolated) < 5:
        return interpolated

    pts = np.array(interpolated)
    x, y = pts[:, 0], pts[:, 1]

    window_length = min(len(interpolated), 11)
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(window_length, 3)

    try:
        x_smooth = savgol_filter(x, window_length, 2)
        y_smooth = savgol_filter(y, window_length, 2)
        return np.vstack((x_smooth, y_smooth)).T.astype(np.int32).tolist()
    except ValueError:
        return interpolated

def process_frame(frame: np.ndarray, _) -> np.ndarray:
    global ball_path

    result = model.infer(frame)[0]
    detections = sv.Detections.from_inference(result)

    class_names = detections.data.get("class_name", np.array([]))
    player_detections = detections[class_names == "player"]
    ball_detections = detections[class_names == "ball"]

    player_detections = tracker.update_with_detections(player_detections)
    annotated_frame = frame.copy()

    if len(ball_detections) > 0:
        x1, y1, x2, y2 = ball_detections.xyxy[0]
        ball_path.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))

        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=ball_detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=ball_detections,
            labels=["ball" for _ in ball_detections],
        )

    if len(ball_path) >= 2:
        smoothed_path = interpolate_and_smooth(list(ball_path))
        for i in range(1, len(smoothed_path)):
            cv2.line(
                annotated_frame,
                tuple(smoothed_path[i - 1]),
                tuple(smoothed_path[i]),
                (0, 255, 255),
                4,
                cv2.LINE_AA,
            )

    if len(player_detections) > 0:
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=player_detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=player_detections,
            labels=["player" for _ in player_detections],
        )

    return annotated_frame

if __name__ == "__main__":
    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TARGET_VIDEO_PATH,
        callback=process_frame,
    )