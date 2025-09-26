# Snap2Console

## The Challenge

<p align="center">
  <img src="https://github.com/ElieSDK/Video_Game_Photo_Recognition/blob/main/figures/game.png" alt="Game Image" />
</p>


I took this picture at Book-Off. Video game covers aren’t always easy to recognize. Titles are often in Japanese, stickers can cover parts of the artwork, and sometimes the Japanese and Western covers look completely different.

The goal of this project was to build a model capable of identifying the game from any uploaded image and providing detailed information about it.

---

## Folder Structure

```
Video_Game_Photo_Recognition/
│
├─ data/                  # Database for EDA
│
├─ notebooks/             # Jupyter notebook for EDA
│   └─ eda.ipynb
│
├─ scripts/               # Python scripts for training and evaluation
│   └─ train.py
│
├─ figures/               # Visualizations
│   └─ confusion_matrix_b2.png
│
├─ requirements.txt       # Python dependencies
└─ README.md

```

## Input Data

The input images are Japanese front covers from [EmuMovies](https://emumovies.com/).

### Games Distribution

| Console | Number of Games |
|---------|----------------|
| Sony Playstation | 3910 |
| Nintendo Super Famicom | 1874 |
| Nintendo Famicom | 1224 |
| Sega Saturn | 1140 |
| Sega Genesis | 458 |
| NEC PC-Engine | 373 |

---

## Exploratory Data Analysis (EDA)

The database used for EDA comes from [IGDB](https://www.igdb.com/) and has more than 230,000 games.

- EDA notebooks are located in the `notebooks` folder.
- The database files are located in the `data` folder.

---

## Training

Training scripts are located in the `scripts` folder.

- The model used for console recognition is **EfficientNet_B2**.
- Training achieved a console recognition accuracy of **95.77%**.

### Confusion Matrix

<p align="center">
  <img src="https://github.com/ElieSDK/Video_Game_Photo_Recognition/blob/main/figures/confusion_matrix_b2.png" alt="Confusion Matrix" width="400"/>
</p>



### Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| FC    | 0.88      | 0.84   | 0.86     | 238     |
| MD    | 0.87      | 0.67   | 0.76     | 101     |
| PCE   | 1.00      | 0.85   | 0.92     | 81      |
| PSX   | 1.00      | 1.00   | 1.00     | 740     |
| SFC   | 0.85      | 0.95   | 0.90     | 401     |
| SS    | 1.00      | 1.00   | 1.00     | 236     |

**Comment:**
The model performs very well on major consoles like PSX and SS, with perfect recall and F1-scores. Some smaller consoles (e.g., MD) show lower recall, which suggests the model struggles when fewer training examples are available.

---

## Game Recognition

For game recognition, we built a **console galleries** using k-NN per console.

- Test results:
  - Console Accuracy: **98.63%**
  - Game Accuracy: **20.35%**

**Explanation:**
- Console accuracy improved because k-NN is trained on top of the EfficientNet_B2 embeddings, enabling better discrimination within each console.
- Game accuracy is probably low because we only have one picture per game.

**Improvement Ideas:**
- Scrape additional game images from the web to increase dataset size.
- Data augmentation was tested but did not improve game accuracy significantly; more images per game are probably required.

---

## Models

I have uploaded the trained models here for download: [Hugging Face Repository](https://huggingface.co/esdk/vg_b2/tree/main)

- `best_console_b2.pth` → Trained EfficientNet_B2 model for console recognition.
- `console_galleries.pkl` → k-NN embeddings for game recognition, organized per console.

**Usage:**
- `best_console_b2.pth` is used to predict the console type of an uploaded image.
- `console_galleries.pkl` is used to find the nearest games within the predicted console using k-NN.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ElieSDK/Snap2Console.git
cd Snap2Console
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

- Place your game images in the `data/` folder.
- The EDA notebooks will read from `data/` and create visualizations.

### 4. Run EDA

```bash
jupyter notebook notebooks/eda.ipynb
```

### 5. Train Console Recognition Model

```bash
python scripts/train.py
```

- The script will save the best model as `best_console_b2.pth`.

### 6. Game Recognition

- Use the console galleries from `console_galleries.pkl` to find nearest games with k-NN.
