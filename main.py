import cv2
from ultralytics import YOLO

def detect_objects_in_video(video_path, output_path='output.mp4'):
    # 1. Load the Model
    # Using our custom trained weights
    # Note: Update this path if you move the model file
    model_path = 'runs/detect/custom_model/weights/best.pt'
    
    try:
        model = YOLO(model_path)
    except:
        print(f"Warning: Custom model not found at {model_path}. Using standard yolov8n.pt.")
        model = YOLO('yolov8n.pt')

    # 2. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing: {video_path} -> {output_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Detect Objects
        results = model(frame)

        # 4. Draw Bounding Boxes
        annotated_frame = results[0].plot()

        # Save Frame
        out.write(annotated_frame)

        # Display Frame
        cv2.imshow('YOLOv8 Object Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing Completed!")

if __name__ == '__main__':
    video_file = "classroom.mp4" 
    detect_objects_in_video(video_file)
