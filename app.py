import numpy as np
import joblib
from flask import Flask, jsonify, render_template
import threading
import time

app = Flask(__name__)

# Load model and data
pipeline = joblib.load('model.pkl')
le = joblib.load('label_encoder.pkl')
X = np.load('X.npy')

# Shared state — the "current" prediction
state = {'label': 'fist', 'confidence': 0.0, 'epoch_index': 0}
state_lock = threading.Lock()

def replay_loop():
    """Cycles through test epochs every 3 seconds, simulating live BCI."""
    idx = 0
    while True:
        epoch = X[idx:idx+1]  # shape (1, 64, 641)
        pred = pipeline.predict(epoch)[0]
        label = le.inverse_transform([pred])[0]
        label_name = 'fist' if label == 2 else 'feet'

        with state_lock:
            state['label'] = label_name
            state['epoch_index'] = idx

        idx = (idx + 1) % len(X)
        time.sleep(3)

# Start background replay thread
t = threading.Thread(target=replay_loop, daemon=True)
t.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/state')
def get_state():
    with state_lock:
        return jsonify(state)

if __name__ == '__main__':
    app.run(debug=False)
