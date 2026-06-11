import os
import random
import shutil
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = r"Face_Model\DataSet_Face"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

for split in ["train", "val", "test"]:

    os.makedirs(
        os.path.join(DATASET_PATH, split),
        exist_ok=True
    )

# ============================================================
# FIND CLASSES
# ============================================================

classes = []

for item in os.listdir(DATASET_PATH):

    item_path = os.path.join(
        DATASET_PATH,
        item
    )

    if (
        os.path.isdir(item_path)
        and item not in ["train", "val", "test"]
    ):
        classes.append(item)

print("Classes Found:")
print(classes)

# ============================================================
# SPLIT DATA
# ============================================================

for class_name in classes:

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    print("\nClass Path:")
    print(class_path)

    print("\nFirst 5 Files:")
    print(os.listdir(class_path)[:5])

    images = []
    print("\nImages Found:")
    print(len(images))

    for file in os.listdir(class_path):

        if os.path.isfile(
            os.path.join(class_path, file)
        ):
            images.append(file)

    print(f"Total Images: {len(images)}")

    # --------------------------------------------------------
    # TRAIN SPLIT
    # --------------------------------------------------------

    train_files, temp_files = train_test_split(
        images,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # --------------------------------------------------------
    # VAL / TEST SPLIT
    # --------------------------------------------------------

    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.50,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    print(
        f"Train: {len(train_files)} | "
        f"Val: {len(val_files)} | "
        f"Test: {len(test_files)}"
    )

    # --------------------------------------------------------
    # CREATE CLASS FOLDERS
    # --------------------------------------------------------

    train_class_dir = os.path.join(
        DATASET_PATH,
        "train",
        class_name
    )

    val_class_dir = os.path.join(
        DATASET_PATH,
        "val",
        class_name
    )

    test_class_dir = os.path.join(
        DATASET_PATH,
        "test",
        class_name
    )

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # --------------------------------------------------------
    # COPY TRAIN
    # --------------------------------------------------------

    for file in train_files:

        shutil.move(
            os.path.join(class_path, file),
            os.path.join(train_class_dir, file)
        )

    # --------------------------------------------------------
    # COPY VAL
    # --------------------------------------------------------

    for file in val_files:

        shutil.move(
            os.path.join(class_path, file),
            os.path.join(val_class_dir, file)
        )

    # --------------------------------------------------------
    # COPY TEST
    # --------------------------------------------------------

    for file in test_files:

        shutil.move(
            os.path.join(class_path, file),
            os.path.join(test_class_dir, file)
        )

print("\nDataset Split Complete!")