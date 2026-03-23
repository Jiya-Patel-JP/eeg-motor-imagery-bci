# EEG Motor Imagery BCI Demo

A Brain-Computer Interface (BCI) project that classifies imagined hand movements from EEG signals and visualises the output in a real-time WebXR scene.

## What it does

- Loads real EEG data from the [PhysioNet Motor Imagery dataset](https://physionet.org/content/eegmmidb/1.0.0/)
- Preprocesses raw brain signals: bandpass filtering (8–30 Hz), epoching
- Classifies **imagined fist vs. feet movement** using a CSP + SVM pipeline
- Replays classifications through a Flask server into a live WebXR scene — the orb changes colour based on the predicted mental state

**Model accuracy: ~89% (5-fold cross-validation)**

## Demo

| Fist | Feet |
|------|------|
| 🔵 Blue orb | 🟢 Green orb |

## Project structure

```
eeg_bci/
├── preprocess.py      # Load EEG data, filter, epoch, save X.npy + y.npy
├── classify.py        # Train CSP + SVM pipeline, evaluate, save model.pkl
├── app.py             # Flask server — replays epochs, serves WebXR frontend
├── templates/
│   └── index.html     # A-Frame WebXR scene
└── requirements.txt
```

## Setup & run

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Step 1 — preprocess (downloads ~7MB of EEG data on first run)
python preprocess.py

# Step 2 — train classifier
python classify.py

# Step 3 — run the demo
python app.py
# Open http://127.0.0.1:5000 in Chrome
```

## Stack

- **MNE-Python** — EEG signal processing
- **scikit-learn** — CSP spatial filtering + SVM classification
- **Flask** — lightweight backend serving predictions
- **A-Frame (WebXR)** — 3D browser scene reacting to BCI output

## Dataset

Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004). BCI2000: A General-Purpose Brain-Computer Interface System. IEEE Transactions on Biomedical Engineering.