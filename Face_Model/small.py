import os

DATASET_PATH = r"Face_Model\DataSet_Face"

for cls in os.listdir(DATASET_PATH):

    cls_path = os.path.join(DATASET_PATH, cls)

    if os.path.isdir(cls_path):

        print(
            cls,
            len(os.listdir(cls_path))
        )