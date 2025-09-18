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

# Define classes and mapping
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

#EfficientNet-B2
num_classes = len(classes)

base_model = EfficientNetB2(weights="imagenet", include_top=False,
                            input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

#Checkpoint
checkpoint = ModelCheckpoint("best_efficientnetb2.h5", 
                             monitor="val_accuracy",
                             save_best_only=True, 
                             mode="max", verbose=1)

#Train
history = model.fit(
    train_images, train_labels,
    epochs=10,
    batch_size=32,
    validation_data=(test_images, test_labels),
    callbacks=[checkpoint]
)

#Evaluate
best_model = load_model("best_efficientnetb2.h5")
test_loss, test_acc = best_model.evaluate(test_images, test_labels)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

#Preediction
img_path = "test_cover.jpg"
img = cv2.imread(img_path)
if img is not None:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img)  # must match training
    img = np.expand_dims(img, axis=0)
    
    pred = best_model.predict(img)
    pred_class = classes[np.argmax(pred)]
    print("Predicted game:", pred_class)
else:
    print(f"Image not found: {img_path}")
