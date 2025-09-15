import cv2
import glob, os, shutil
import numpy as np

folder_path = "/home/es/Documents/Project/VG/Medias/Nintendo Famicom (2D Boxes)(Kondorito 1.2)"

# List all files in the folder
images = os.listdir(folder_path)

#Create an individual folder for each file
folder = "/home/es/Documents/Project/VG/Medias/Nintendo Famicom (2D Boxes)(Kondorito 1.2)"

for file_path in glob.glob(os.path.join(folder, '*.*')):
    new_dir = file_path.rsplit('.', 1)[0]
    os.mkdir(os.path.join(folder, new_dir))
    shutil.move(file_path, os.path.join(new_dir, os.path.basename(file_path)))


#I will probably use EfficientNet-B2 because our dataset is small (3k pictures) as ResNet-50 / VGG16... require a bigger dataset.

IMG_SIZE = 256 #EfficientNet-B2 supported resolution
DATA_DIR = "/home/es/Documents/Project/VG/Medias/Nintendo Famicom (2D Boxes)(Kondorito 1.2)"

images = []
labels = []

classes = os.listdir(DATA_DIR)  # Each folder is a class
class_dict = {cls_name: idx for idx, cls_name in enumerate(classes)}

for cls_name in classes:
    cls_path = os.path.join(DATA_DIR, cls_name)
    for img_file in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_file)
        img = cv2.imread(img_path)       # Read image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # Resize
        images.append(img)
        labels.append(class_dict[cls_name])

images = np.array(images) / 255.0  # Normalize
labels = np.array(labels)
