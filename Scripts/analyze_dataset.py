# import os
# import yaml

# def analyze_dataset_structure(dataset_path):
#     """
#     Analyzes the structure of a Roboflow-style dataset,
#     reads class names from data.yaml, and counts images in splits.
#     """
#     print(f"🔍 Analyzing dataset at: {dataset_path}")

#     data_yaml_path = os.path.join(dataset_path, 'data.yaml')
#     if not os.path.exists(data_yaml_path):
#         print(f"❌ Error: data.yaml not found at {data_yaml_path}")
#         print("Please ensure the path points to the root of your downloaded dataset.")
#         return

#     try:
#         with open(data_yaml_path, 'r') as f:
#             data_config = yaml.safe_load(f)
        
#         names = data_config.get('names', [])
#         nc = data_config.get('nc', len(names)) # nc is number of classes
        
#         print(f"\n📋 Dataset Configuration (from data.yaml):")
#         print(f"  Number of classes (nc): {nc}")
#         print(f"  Class Names (names): {names}")

#         splits = ['train', 'valid', 'test']
#         print("\n📊 Image Counts per Split and Class:")
        
#         total_images = 0
#         class_counts = {name: 0 for name in names}

#         for split in splits:
#             split_path = os.path.join(dataset_path, split)
#             if not os.path.exists(split_path):
#                 print(f"  ⚠️ Warning: '{split}' directory not found at {split_path}")
#                 continue
            
#             print(f"\n--- {split.upper()} ---")
#             split_total = 0
            
#             # Assuming images are directly in the split folder, or in subfolders per class
#             # For Roboflow, images are often directly in 'train/images', 'valid/images', etc.
#             # Let's check for 'images' subfolder first
#             images_path = os.path.join(split_path, 'images')
#             if not os.path.exists(images_path):
#                 images_path = split_path # Fallback if no 'images' subfolder

#             for class_name in names:
#                 class_dir_path = os.path.join(images_path, class_name)
#                 count = 0
#                 if os.path.exists(class_dir_path) and os.path.isdir(class_dir_path):
#                     # If classes are in subfolders
#                     for root, _, files in os.walk(class_dir_path):
#                         for file in files:
#                             if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
#                                 count += 1
#                 else:
#                     # If images are flat in 'images_path' and labels are separate
#                     # This is more common for detection datasets, but for classification,
#                     # images are usually in class subfolders.
#                     # For now, we'll just count all images in the split's image folder.
#                     # A more robust check would involve parsing labels to count per class.
#                     pass # We'll rely on the class subfolder structure for now.

#                 if count > 0:
#                     print(f"  {class_name}: {count} images")
#                     class_counts[class_name] += count
#                     split_total += count
            
#             # Fallback: if no class subfolders, just count all images in the split's image folder
#             if split_total == 0:
#                 all_images_in_split = 0
#                 for root, _, files in os.walk(images_path):
#                     for file in files:
#                         if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
#                             all_images_in_images_path += 1
#                 if all_images_in_images_path > 0:
#                     print(f"  Total images in {split} split (no class subfolders detected): {all_images_in_images_path}")
#                     split_total = all_images_in_images_path

#             print(f"  Total images in {split} split: {split_total}")
#             total_images += split_total

#         print(f"\n--- Overall Statistics ---")
#         print(f"Total images across all splits: {total_images}")
#         print(f"Overall class distribution:")
#         for name, count in class_counts.items():
#             print(f"  {name}: {count} images")

#     except Exception as e:
#         print(f"❌ An error occurred during analysis: {e}")

# if __name__ == "__main__":
#     # IMPORTANT: Replace 'path/to/your/downloaded_dataset' with the actual path
#     # where you extracted your dataset on your local machine.
#     # Example: analyze_dataset_structure('C:/Users/YourUser/Downloads/Kaggle_tomato_pest_dataset')
#     # Example: analyze_dataset_structure('/home/user/datasets/Kaggle_tomato_pest_dataset')
    
#     # You will need to manually update this path after downloading the dataset.
#     dataset_root_path = "C:/Users/taofe/Desktop/Kaggle tomato pest dataset" 
    
#     # Ensure you have PyYAML installed: pip install PyYAML
    
#     analyze_dataset_structure(dataset_root_path)


import os
import yaml

def analyze_dataset_structure(dataset_path):
    """
    Analyzes the structure of a Roboflow-style dataset,
    reads class names from data.yaml, and counts images in splits.
    """
    print(f"🔍 Analyzing dataset at: {dataset_path}")

    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(data_yaml_path):
        print(f"❌ Error: data.yaml not found at {data_yaml_path}")
        print("Please ensure the path points to the root of your downloaded dataset.")
        return

    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        names = data_config.get('names', [])
        nc = data_config.get('nc', len(names)) # nc is number of classes
        
        print(f"\n📋 Dataset Configuration (from data.yaml):")
        print(f"  Number of classes (nc): {nc}")
        print(f"  Class Names (names): {names}")

        splits = ['train', 'valid', 'test']
        print("\n📊 Image Counts per Split and Class:")
        
        total_images_overall = 0 # Renamed to avoid confusion with split_total
        class_counts_overall = {name: 0 for name in names}

        for split in splits:
            split_path = os.path.join(dataset_path, split)
            if not os.path.exists(split_path):
                print(f"  ⚠️ Warning: '{split}' directory not found at {split_path}")
                continue
            
            print(f"\n--- {split.upper()} ---")
            split_total = 0
            
            # For Roboflow, images are often directly in 'train/images', 'valid/images', etc.
            images_path = os.path.join(split_path, 'images')
            if not os.path.exists(images_path):
                images_path = split_path # Fallback if no 'images' subfolder (e.g., PlantVillage)

            # Attempt to count images within class subfolders first (typical for classification)
            found_class_subfolders = False
            for class_name in names:
                class_dir_path = os.path.join(images_path, class_name)
                count = 0
                if os.path.exists(class_dir_path) and os.path.isdir(class_dir_path):
                    found_class_subfolders = True
                    for root, _, files in os.walk(class_dir_path):
                        for file in files:
                            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                                count += 1
                    if count > 0:
                        print(f"  {class_name}: {count} images")
                        class_counts_overall[class_name] += count
                        split_total += count
            
            # Fallback: if no class subfolders were found, just count all images in the split's image folder
            if not found_class_subfolders:
                all_images_in_images_path = 0 # Initialize the variable here
                for root, _, files in os.walk(images_path):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                            all_images_in_images_path += 1
                if all_images_in_images_path > 0:
                    print(f"  Total images in {split} split (no class subfolders detected): {all_images_in_images_path}")
                    split_total = all_images_in_images_path
                    print("  Note: Class-specific counts may not be accurate if images are not in class subfolders.")

            print(f"  Total images in {split} split: {split_total}")
            total_images_overall += split_total

        print(f"\n--- Overall Statistics ---")
        print(f"Total images across all splits: {total_images_overall}")
        print(f"Overall class distribution:")
        for name, count in class_counts_overall.items():
            print(f"  {name}: {count} images")

    except Exception as e:
        print(f"❌ An error occurred during analysis: {e}")

if __name__ == "__main__":
    # IMPORTANT: Replace 'path/to/your/downloaded_dataset' with the actual path
    # where you extracted your dataset on your local machine.
    # Example: analyze_dataset_structure(r'C:\Users\YourUser\Downloads\Kaggle_tomato_pest_dataset')
    # Example: analyze_dataset_structure('/home/user/datasets/Kaggle_tomato_pest_dataset')
    
    # You will need to manually update this path after downloading the dataset.
    dataset_root_path = r"C:\Users\taofe\Desktop\Kaggle tomato pest dataset" # <--- ENSURE THIS IS CORRECT
    
    # Ensure you have PyYAML installed: pip install PyYAML
    
    analyze_dataset_structure(dataset_root_path)
