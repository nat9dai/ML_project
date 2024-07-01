import os

# Define the mapping from the original class to the new hierarchical class
class_mapping = {
    "Cardboard": "Recyclable",
    "Food Organics": "Organic",
    "Glass": "Recyclable",
    "Metal": "Recyclable",
    "Miscellaneous Trash": "Non-Recyclable",
    "Paper": "Recyclable",
    "Plastic": "Non-Recyclable",
    "Textile Trash": "Non-Recyclable",
    "Vegetation": "Organic"
}

# Define the level 1 classes
level_1_classes = ["Organic", "Recyclable", "Non-Recyclable"]

# Path to the dataset directory
dataset_path = "/home/nat/DL_in_practice/ML_project/dataset-realwaste-hierarchical/Recyclable/Recyclable-Paper/"
name = "Recyclable-Paper"
file_type = "jpg"

count = 1
for file in os.listdir(dataset_path):
    old_file_path = os.path.join(dataset_path, file)
    new_file_name = f"{name}-{str(count)}.{file_type}"

    new_file_path = os.path.join(dataset_path, new_file_name)
    os.rename(old_file_path, new_file_path)
    count +=1

exit()

# Iterate over the classes in the dataset
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)

    # Skip if it's not a directory
    if not os.path.isdir(class_path):
        continue

    # Process only level 2 classes
    #if class_name not in level_1_classes:
    if (True):
        # Rename files within each class directory
        count = 1
        for file_name in os.listdir(class_path):
            old_file_path = os.path.join(class_path, file_name)
            # Replace underscores with hyphens in the file name
            file_name_with_hyphens = file_name.replace("_", "-")
            new_file_name = f"{class_mapping[class_name]}-{class_name}-{str(count)}"
            new_file_path = os.path.join(class_path, new_file_name)

            # Rename the file
            os.rename(old_file_path, new_file_path)
            count += 1

        # Rename the class directory
        #new_class_name = f"{class_mapping[class_name]}-{class_name}"
        #new_class_path = os.path.join(dataset_path, new_class_name)
        #os.rename(class_path, new_class_path)

# Note: Level 1 class directories are not renamed
