import os
import shutil
import yaml
from tqdm import tqdm

def convert_detection_to_classification(source_dataset_path, target_dataset_path):
    """
    Converts an object detection dataset (Roboflow YOLO format) into an image classification dataset.
    Each image is assigned to the first class found in its label file.
    """
    print(f"🔄 Converting dataset from detection to classification format...")
    print(f"   Source: {source_dataset_path}")
    print(f"   Target: {target_dataset_path}")

    # Load data.yaml to get class names
    data_yaml_path = os.path.join(source_dataset_path, 'data.yaml')
    if not os.path.exists(data_yaml_path):
        print(f"❌ Error: data.yaml not found at {data_yaml_path}")
        print("Please ensure the path points to the root of your downloaded dataset.")
        return

    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config.get('names', [])
    if not class_names:
        print("❌ Error: No class names found in data.yaml. Cannot proceed.")
        return

    print(f"\n📋 Detected classes: {class_names}")

    splits = ['train', 'valid', 'test']
    total_converted_images = 0

    for split in splits:
        source_images_dir = os.path.join(source_dataset_path, split, 'images')
        source_labels_dir = os.path.join(source_dataset_path, split, 'labels')
        
        if not os.path.exists(source_images_dir) or not os.path.exists(source_labels_dir):
            print(f"⚠️ Warning: Skipping '{split}' split. Missing 'images' or 'labels' directory at {os.path.join(source_dataset_path, split)}.")
            continue

        print(f"\n--- Processing {split.upper()} split ---")
        
        # Create target directories for this split (e.g., classification_dataset_for_vit/train/Aphid)
        for class_name in class_names:
            os.makedirs(os.path.join(target_dataset_path, split, class_name), exist_ok=True)

        image_files = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        for img_file in tqdm(image_files, desc=f"Converting {split} images"):
            img_name_without_ext = os.path.splitext(img_file)[0]
            label_file = img_name_without_ext + '.txt'
            label_path = os.path.join(source_labels_dir, label_file)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                if lines:
                    # For simplicity, assign to the first detected class in the label file
                    try:
                        first_line = lines[0].strip().split(' ')
                        class_id = int(first_line[0])
                        
                        if 0 <= class_id < len(class_names):
                            target_class_name = class_names[class_id]
                            source_img_path = os.path.join(source_images_dir, img_file)
                            target_img_path = os.path.join(target_dataset_path, split, target_class_name, img_file)
                            
                            shutil.copy2(source_img_path, target_img_path)
                            total_converted_images += 1
                        else:
                            print(f"⚠️ Warning: Invalid class ID {class_id} in {label_file}. Skipping image {img_file}.")
                    except (ValueError, IndexError) as e:
                        print(f"⚠️ Warning: Error parsing label file {label_file}: {e}. Skipping image {img_file}.")
                else:
                    print(f"⚠️ Warning: Label file {label_file} is empty. Skipping image {img_file}.")
            else:
                print(f"⚠️ Warning: No label file found for {img_file}. Skipping image.")
    
    print(f"\n✅ Conversion complete! Total images converted: {total_converted_images}")
    print(f"The new classification dataset is located at: {target_dataset_path}")
    print(f"You can now use '{target_dataset_path}' as your DATA_DIR for ViT training.")

if __name__ == "__main__":
    # IMPORTANT: Update these paths
    # This should be the root of your downloaded object detection dataset (e.g., where train, valid, test, data.yaml are)
    SOURCE_DATASET_PATH = r"C:\Users\taofe\Desktop\Kaggle tomato pest dataset" 
    
    # This will be the new directory where the classification-ready dataset will be created
    TARGET_DATASET_PATH = "classification_dataset_for_vit" 
    
    convert_detection_to_classification(SOURCE_DATASET_PATH, TARGET_DATASET_PATH)
