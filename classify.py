import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from mne.decoding import CSP
import joblib

# Load preprocessed data
X = np.load('X.npy')   # shape: (45, 64, 641)
y = np.load('y.npy')   # shape: (45,)

# Encode labels to 0/1
le = LabelEncoder()
y = le.fit_transform(y)
print("Classes:", le.classes_, "→ encoded as 0 and 1")
print("X shape:", X.shape, "| y shape:", y.shape)

# Build CSP + SVM pipeline
# CSP: finds spatial filters that maximise class separability
# SVM: classifies the filtered features
pipeline = Pipeline([
    ('csp', CSP(n_components=4, reg=None, log=True, norm_trace=False)),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))
])

# Cross-validate
# StratifiedKFold keeps class balance in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

print("\n── Results ───────────────────────────────")
print(f"Fold accuracies: {[f'{s:.2f}' for s in scores]}")
print(f"Mean accuracy:   {scores.mean():.2%}")
print(f"Std deviation:   {scores.std():.2%}")

# Train final model on all data & save
pipeline.fit(X, y)
joblib.dump(pipeline, 'model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("\nModel saved to model.pkl")
