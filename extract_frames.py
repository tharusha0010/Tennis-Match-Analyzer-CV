import cv2
import os

def extract_frames(video_path, output_dir, frame_interval=30):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_name = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1

        count += 1

    cap.release()
    print(f"Extracted {saved_count} frames successfully.")

if __name__ == "__main__":
    extract_frames("input_videos/test_video.mp4", "extracted_frames", frame_interval=30)