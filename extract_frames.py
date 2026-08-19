import cv2
import os

def extract_frames(video_path, output_dir, max_frames=200):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_interval = max(1, total_frames // max_frames)
    
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
            
            if saved_count >= max_frames:
                break

        count += 1

    cap.release()
    print(f"Total frames in video: {total_frames}")
    print(f"Extracted {saved_count} frames successfully to {output_dir}.")

if __name__ == "__main__":
    extract_frames("input_videos/test_video.mp4", "extracted_frames/new_frames", max_frames=200)