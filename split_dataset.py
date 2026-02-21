import os
import shutil
import random

# Paths
field_path = "dataset/images/field"
lab_path = "dataset/images/lab"

train_path = "dataset/train"
test_path = "dataset/test"

# Create train/test directories
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

species_list = os.listdir(field_path)

for species in species_list:
    print(f"Processing {species}...")

    # Create species folders
    os.makedirs(os.path.join(train_path, species), exist_ok=True)
    os.makedirs(os.path.join(test_path, species), exist_ok=True)

    # Collect images from field + lab
    field_images = os.listdir(os.path.join(field_path, species))
    lab_images = os.listdir(os.path.join(lab_path, species))

    all_images = []

    for img in field_images:
        all_images.append(os.path.join(field_path, species, img))

    for img in lab_images:
        all_images.append(os.path.join(lab_path, species, img))

    # Shuffle images
    random.shuffle(all_images)

    # Split 80-20
    split_index = int(0.8 * len(all_images))

    train_images = all_images[:split_index]
    test_images = all_images[split_index:]

    # Copy images
    for img_path in train_images:
        shutil.copy(img_path, os.path.join(train_path, species))

    for img_path in test_images:
        shutil.copy(img_path, os.path.join(test_path, species))

print("Dataset split completed successfully!")
