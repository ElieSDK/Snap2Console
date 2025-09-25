import os
import pickle
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Dataset
class GameDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.transform = transform
        self.images, self.console_labels, self.game_labels = [], [], []
        for f in os.listdir(folder):
            if f.lower().endswith((".jpg",".png",".jpeg")):
                self.images.append(os.path.join(folder, f))
                console = f.split("_")[0]
                game = f[len(console)+1:].rsplit('.',1)[0]
                self.console_labels.append(console)
                self.game_labels.append(game)
        self.consoles = sorted(set(self.console_labels))
        self.console_to_idx = {c:i for i,c in enumerate(self.consoles)}
        self.idx_to_console = {i:c for c,i in self.console_to_idx.items()}
        self.console_idx = [self.console_to_idx[c] for c in self.console_labels]

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if img.mode in ("P", "LA") or "transparency" in img.info:
            img = img.convert("RGBA").convert("RGB")
        else:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.console_idx[idx], self.game_labels[idx]

# Load data
data_dir = "C:/Users/PC/Desktop/vg/Medias"
dataset = GameDataset(data_dir, transform=train_transform)
num_consoles = len(dataset.consoles)
num_test = int(len(dataset)*0.2)
train_ds, test_ds = random_split(dataset, [len(dataset)-num_test, num_test])
test_ds.dataset.transform = test_transform

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

# Model
model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features if isinstance(model.classifier, nn.Sequential) else model.classifier.in_features
model.classifier = nn.Linear(in_features, num_consoles)
model = model.to(device)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

best_acc = 0
for epoch in range(5):
    model.train()
    for images, consoles, _ in train_loader:
        images, consoles = images.to(device), consoles.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, consoles)
        loss.backward()
        optimizer.step()

    # Eval
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, consoles, _ in test_loader:
            images, consoles = images.to(device), consoles.to(device)
            preds = model(images).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(consoles.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1} - Console Accuracy: {acc*100:.2f}%")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_console_b2.pth")
        print("Best model saved")
        

# Evaluation
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, consoles, _ in test_loader:
        images, consoles = images.to(device), consoles.to(device)
        preds = model(images).argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(consoles.cpu().numpy())

# Accuracy
acc = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {acc*100:.2f}%\n")

# Console names
console_names = dataset.consoles

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=console_names,
            yticklabels=console_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Console Confusion Matrix")
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=console_names))


# Backbone for embeddings (EfficientNet-B2 without classifier)
backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
backbone.classifier = nn.Identity()
backbone = backbone.to(device)
backbone.eval()

# Build console galleries (k-NN for games)
console_galleries = {}
for img_path, console_idx, game_label in tqdm(zip(dataset.images, dataset.console_idx, dataset.game_labels), total=len(dataset)):
    img = Image.open(img_path)
    if img.mode in ("P", "LA") or "transparency" in img.info:
        img = img.convert("RGBA").convert("RGB")
    else:
        img = img.convert("RGB")
    img_t = test_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = backbone(img_t).detach().cpu().numpy().flatten()
        emb /= np.linalg.norm(emb)
    console_name = dataset.idx_to_console[console_idx]
    if console_name not in console_galleries:
        console_galleries[console_name] = {"embs": [], "games": []}
    console_galleries[console_name]["embs"].append(emb)
    console_galleries[console_name]["games"].append(game_label)

# Fit k-NN per console
k = 5
for console in console_galleries:
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(console_galleries[console]["embs"])
    console_galleries[console]["knn"] = knn

print("k-NN galleries ready")

# Evaluate Console + Game
y_true_console, y_pred_console = [], []
y_true_game, y_pred_game = [], []

model.eval()
for img_path, console_idx, game_label in tqdm(zip(test_ds.dataset.images, test_ds.dataset.console_idx, test_ds.dataset.game_labels), total=len(test_ds)):
    img = Image.open(img_path)
    if img.mode in ("P", "LA") or "transparency" in img.info:
        img = img.convert("RGBA").convert("RGB")
    else:
        img = img.convert("RGB")
    img_t = test_transform(img).unsqueeze(0).to(device)

    # Predict console
    with torch.no_grad():
        console_logits = model(img_t)
        pred_console_idx = console_logits.argmax(1).item()
        pred_console_name = dataset.idx_to_console[pred_console_idx]

    # Predict game with k-NN
    emb = backbone(img_t).detach().cpu().numpy().flatten()
    emb /= np.linalg.norm(emb)
    knn = console_galleries[pred_console_name]["knn"]
    distances, indices = knn.kneighbors([emb])
    votes = [console_galleries[pred_console_name]["games"][i] for i in indices[0]]
    pred_game = max(set(votes), key=votes.count)

    # Store results
    y_true_console.append(console_idx)
    y_pred_console.append(pred_console_idx)
    y_true_game.append(game_label)
    y_pred_game.append(pred_game)

# Metrics
console_acc = accuracy_score(y_true_console, y_pred_console)
game_acc = accuracy_score(y_true_game, y_pred_game)
print(f"\n📊 Test Console Accuracy: {console_acc*100:.2f}%")
print(f"📊 Test Game Accuracy: {game_acc*100:.2f}%")


with open("console_galleries.pkl", "wb") as f:
    pickle.dump(console_galleries, f)