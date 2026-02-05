import os
import shutil
import random

def split_dataset(val_ratio=0.2):
    # Source paths
    images_source = "dataset/raw_images"
    labels_source = "dataset/auto_labels"
    
    # Destination paths
    base_dir = "dataset"
    train_images = os.path.join(base_dir, "train", "images")
    train_labels = os.path.join(base_dir, "train", "labels")
    val_images = os.path.join(base_dir, "val", "images")
    val_labels = os.path.join(base_dir, "val", "labels")
    
    # Create directories
    for folder in [train_images, train_labels, val_images, val_labels]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Get file list
    files = [f for f in os.listdir(images_source) if f.endswith(('.jpg', '.png'))]
    random.shuffle(files)
    
    val_count = int(len(files) * val_ratio)
    
    print(f"Total images: {len(files)}")
    print(f"Training: {len(files) - val_count}, Validation: {val_count}")
    
    for i, filename in enumerate(files):
        # Source paths
        src_image_path = os.path.join(images_source, filename)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        src_label_path = os.path.join(labels_source, txt_filename)
        
        # Determine destination
        if i < val_count:
            dst_image_folder = val_images
            dst_label_folder = val_labels
        else:
            dst_image_folder = train_images
            dst_label_folder = train_labels
            
        # Copy image
        shutil.copy(src_image_path, os.path.join(dst_image_folder, filename))
        
        # Copy label or create empty one
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, os.path.join(dst_label_folder, txt_filename))
        else:
            # Create empty text file for images with no objects
            with open(os.path.join(dst_label_folder, txt_filename), 'w') as f:
                pass
                
    print("Dataset split completed successfully!")
    print(f"Structure: {base_dir}/train and {base_dir}/val")

if __name__ == "__main__":
    split_dataset()
