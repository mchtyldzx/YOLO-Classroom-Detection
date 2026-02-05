from ultralytics import YOLO
import os

def auto_label_images():
    # Paths
    images_dir = "dataset/raw_images"
    labels_dir = "dataset/auto_labels"
    
    if not os.path.exists(images_dir):
        print(f"Error: Directory '{images_dir}' not found. Run extract_frames.py first.")
        return

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # Load pre-trained model (yolov8n is fast and good for standard objects)
    print("Loading model...")
    model = YOLO('yolov8n.pt') 

    print("Starting auto-labeling (looking for 'person', 'table', etc.)...")
    
    # Iterate over all images
    for filename in os.listdir(images_dir):
        if not filename.endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(images_dir, filename)
        
        # Run inference
        results = model.predict(image_path, conf=0.4, verbose=False)
        
        # Prepare label file name
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(labels_dir, txt_filename)

        with open(txt_path, "w") as f:
            for result in results:
                for box in result.boxes:
                    # COCO Class mapping: 
                    # 0: person
                    # 56: chair (mapping to desk)
                    # 60: dining table (mapping to table)
                    
                    coco_class_id = int(box.cls[0])
                    x, y, w, h = box.xywhn[0]  # Normalized coordinates
                    
                    # Map COCO classes to our custom classes
                    # Our classes: 0: person, 1: desk, 2: table, 3: window
                    user_class_id = -1
                    
                    if coco_class_id == 0:  # Person
                        user_class_id = 0
                    elif coco_class_id == 60: # Table
                        user_class_id = 2
                    elif coco_class_id == 56: # Chair -> Desk
                         user_class_id = 1
                    
                    # If the object is one of interest, write to file
                    if user_class_id != -1:
                        # Format: class_id center_x center_y width height
                        line = f"{user_class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                        f.write(line)

        print(f"Labeled: {txt_filename}")

    print("\nCompleted!")
    print(f"Labels saved to '{labels_dir}'.")
    print("NOTE: 'Window' objects are not auto-detected by standard COCO model. Manual addition might be needed.")

if __name__ == "__main__":
    auto_label_images()
