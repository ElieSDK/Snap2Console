import cv2
import glob, os, shutil
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths
DATA_DIR = "/home/es/Documents/Project/VG/Medias/Nintendo Famicom (2D Boxes)(Kondorito 1.2)"
IMG_SIZE = 260  # EfficientNet-B2 compatible

# Organize images into folders
'''
Example of filename:
GB_Tetris.jpg  
SFC_Tetris.jpg  
NES_Mario.jpg 
'''
for file_path in glob.glob(os.path.join(DATA_DIR, '*.*')):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    new_dir = os.path.join(DATA_DIR, base_name)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    shutil.move(file_path, os.path.join(new_dir, os.path.basename(file_path)))

# Load images
# Re-preprocess images for EfficientNet
images = []
labels = []

# 🔹 Define classes and mapping
classes = sorted(os.listdir(DATA_DIR))  # Each folder is a class
class_dict = {cls_name: idx for idx, cls_name in enumerate(classes)}

for cls_name in classes:
    cls_path = os.path.join(DATA_DIR, cls_name)
    for img_file in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = preprocess_input(img)  # EfficientNet preprocessing
        images.append(img)
        labels.append(class_dict[cls_name])

images = np.array(images)
labels = np.array(labels)

train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=0, stratify=labels
)


