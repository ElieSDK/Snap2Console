import cv2
import glob, os, shutil
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths
DATA_DIR = "/home/es/Documents/Project/VG/Medias/Nintendo Famicom (2D Boxes)(Kondorito 1.2)"
IMG_SIZE = 256  # EfficientNet-B2 compatible

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
images = []
labels = []

classes = sorted(os.listdir(DATA_DIR))  # # Each folder is a class + Ensure consistent order
class_dict = {cls_name: idx for idx, cls_name in enumerate(classes)}

for cls_name in classes:
    cls_path = os.path.join(DATA_DIR, cls_name)
    for img_file in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_file)
        img = cv2.imread(img_path) # Read image
        if img is None:
            continue  # skip broken images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # Resize
        images.append(img) # Adds the matrix to the list of images
        labels.append(class_dict[cls_name])

images = np.array(images) / 255.0 # Normalize
labels = np.array(labels)

# Train/Validation split
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=0, stratify=labels
)

# CNN model using TensorFlow/Keras
num_classes = len(classes)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)), #Extract features from images
    MaxPooling2D((2,2)), #Downsample feature maps
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(), # Convert 2D feature maps to 1D vector for fully connected layers
    Dense(128, activation='relu'), # Combine extracted features to make final classification
    Dropout(0.5), # Regularization to prevent overfitting
    Dense(num_classes, activation='softmax') # Output layer: outputs a probability for each class (game)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Checkpoint
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy', # track validation accuracy
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train model
history = model.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=15,
    batch_size=32,
    callbacks=[checkpoint]
)

# Load and evaluate best model
best_model = load_model('best_model.h5')
val_loss, val_acc = best_model.evaluate(val_images, val_labels)
print("Best Validation loss:", val_loss)
print("Best Validation accuracy:", val_acc)

# Predict
img_path = "test_cover.jpg"
img = cv2.imread(img_path)
if img is not None:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0) # Add batch dimension
    
    pred = best_model.predict(img)
    pred_class = classes[np.argmax(pred)]
    print("Predicted game:", pred_class)
else:
    print(f"Image not found: {img_path}")




# EfficientNet-B2