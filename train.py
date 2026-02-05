from ultralytics import YOLO

def train_custom_model():
    # 1. Initialize Model
    # Using 'yolov8n.pt' (pretrained) as a starting point for transfer learning
    model = YOLO('yolov8n.pt') 

    # 2. Start Training
    # data: Path to the data configuration file
    # epochs: Number of training cycles
    print("Starting training...")
    model.train(
        data='data.yaml', 
        epochs=50, 
        imgsz=640,
        name='custom_model' # Results will be saved to runs/detect/custom_model
    )
    print("Training finished!")

if __name__ == '__main__':
    train_custom_model()
