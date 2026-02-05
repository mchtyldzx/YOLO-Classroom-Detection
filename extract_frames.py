import cv2
import os

# Settings
video_path = "classroom.mp4"       # Input video file
output_folder = "dataset/raw_images" # Folder to save extracted frames
frame_interval = 30                # Capture one frame every N frames (30 = approx 1 sec)

def extract_frames():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file '{video_path}'")
        return

    count = 0
    saved_count = 0

    print("Starting frame extraction...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame at the defined interval
        if count % frame_interval == 0:
            frame_name = os.path.join(output_folder, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1
        
        count += 1

    cap.release()
    print(f"Completed! {saved_count} images saved to '{output_folder}'.")

if __name__ == "__main__":
    if os.path.exists(video_path):
        extract_frames()
    else:
        print(f"ERROR: File '{video_path}' not found. Please place the video in the project directory.")
