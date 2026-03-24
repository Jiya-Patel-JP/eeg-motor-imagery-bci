# Pipeline:
#   1. Load PhysioNet Motor Imagery data (Subject 1, runs 6/10/14)
#   2. Standardise channel names & apply montage
#   3. Bandpass filter: 8–30 Hz (mu + beta band)
#   4. ICA artifact removal
#        - Ocular  : automated via Fp1 proxy (no dedicated EOG channel)
#        - Muscle  : automated via find_bads_muscle
#        - Prints a full diagnostic report for manual verification
#   5. Epoch extraction (T1=fist, T2=feet)
#   6. Bad epoch rejection via peak-to-peak amplitude threshold
#   7. Save X.npy (epochs × channels × times) and y.npy (labels)

import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.preprocessing import ICA

# 0. Reproducibility
RANDOM_STATE = 42
mne.set_log_level("WARNING")   # suppress routine MNE verbosity; we print our own

# 1. Load raw data
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

# Runs 6, 10, 14: imagined fist (T1) and feet (T2) movement
raw_files = [
    read_raw_edf(f, preload=True, stim_channel="auto")
    for f in eegbci.load_data(1, [6, 10, 14])
]
raw = concatenate_raws(raw_files)
print(f"  Channels : {raw.info['nchan']}")
print(f"  Duration : {raw.times[-1]:.1f} s")
print(f"  Sampling : {raw.info['sfreq']} Hz")

# 2. Channel setup
eegbci.standardize(raw)
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage)
print("\n  Montage applied: standard_1005")

# 3. Bandpass filter
print("\n" + "=" * 60)
print("STEP 2: Bandpass filter (8–30 Hz)")
print("=" * 60)
raw.filter(8.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
print("  Filter applied.")

# 4. ICA artifact removal
print("\n" + "=" * 60)
print("STEP 3: ICA artifact removal")
print("=" * 60)

picks_eeg = mne.pick_types(
    raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
)

# n_components=20 captures dominant sources without overfitting on 64-ch data.
# fastica is the standard algorithm for EEG; picard is an alternative but
# requires an extra dependency and offers no meaningful gain here.
ica = ICA(
    n_components=20,
    method="fastica",
    fit_params=dict(extended=True),   # extended infomax variant - handles
                                       # both sub- and super-Gaussian sources,
                                       # giving more stable decompositions
    random_state=RANDOM_STATE,
    max_iter="auto",
)

print("  Fitting ICA (this may take ~30 s)...")
ica.fit(raw, picks=picks_eeg)
print(f"  ICA fitted on {ica.n_components_} components.")

# 4a. Ocular artifact detection
# PhysioNet motor imagery has no dedicated EOG channel.
# Fp1 (frontal pole) is the standard proxy: it sits closest to the eyes and
# captures blink / saccade variance reliably.
# threshold=3.0 z-score: conservative - avoids flagging real neural components.
eog_indices, eog_scores = ica.find_bads_eog(
    raw, ch_name="Fp1", threshold=3.0, measure="zscore"
)

print(f"\n  [Ocular] Components flagged : {eog_indices if eog_indices else 'none'}")
if eog_indices:
    for idx, score in zip(eog_indices, eog_scores[eog_indices]):
        print(f"    IC{idx:02d}  z-score = {score:.3f}")

# 4b. Muscle artifact detection 
# find_bads_muscle identifies high-frequency (>= 7 Hz post-filter) power
# bursts characteristic of EMG contamination.
# threshold=0.5 is MNE's recommended starting point for 64-ch data.
muscle_indices, muscle_scores = ica.find_bads_muscle(raw, threshold=0.5)

print(f"\n  [Muscle] Components flagged : {muscle_indices if muscle_indices else 'none'}")
if muscle_indices:
    for idx, score in zip(muscle_indices, muscle_scores[muscle_indices]):
        print(f"    IC{idx:02d}  score = {score:.3f}")

# 4c. Combine & apply 
# Deduplicate in case a component scored on both detectors
all_bad = sorted(set(eog_indices + muscle_indices))
ica.exclude = all_bad

print(f"\n  Total components excluded : {len(all_bad)}  {all_bad}")
print("  Applying ICA (projecting out artifact components)...")
raw_clean = ica.apply(raw.copy())
print("  ICA applied.")

# 4d. Diagnostic report 
print("\n ICA Diagnostic Report ")
print(f"  Components fitted      : {ica.n_components_}")
print(f"  Ocular  (Fp1 proxy)    : {eog_indices}  "
      f"(z-score threshold = 3.0)")
print(f"  Muscle  (find_bads)    : {muscle_indices}  "
      f"(score threshold = 0.5)")
print(f"  Total removed          : {len(all_bad)}")
print(f"  Variance explained retained: "
      f"{100 * (1 - len(all_bad) / ica.n_components_):.1f}%")
print("  ----------------------------------------------")

# 5. Epoch extraction
print("\n" + "=" * 60)
print("STEP 4: Epoch extraction")
print("=" * 60)

events, event_id = mne.events_from_annotations(raw_clean)
print(f"  Event types found: {event_id}")

picks = mne.pick_types(
    raw_clean.info, meg=False, eeg=True, stim=False, eog=False
)

# tmin=0, tmax=4: use the full 4-second imagery window (standard for PhysioNet)
# baseline=None: no baseline correction — the bandpass filter handles DC offset
epochs = mne.Epochs(
    raw_clean,
    events,
    event_id=dict(fist=2, feet=3),
    tmin=0.0,
    tmax=4.0,
    proj=True,
    picks=picks,
    baseline=None,
    preload=True,
)
print(f"  Epochs before rejection : {len(epochs)}")

# 6. Bad epoch rejection
print("\n" + "=" * 60)
print("STEP 5: Bad epoch rejection (peak-to-peak threshold)")
print("=" * 60)

# ICA removes sustained/recurring artifacts (blinks, muscle bursts)
# Epoch rejection catches residual transient events ICA cannot model
# e.g. electrode pop, sudden movement spike — that would distort CSP
#
# 150 µV is the standard threshold for motor imagery EEG:
#   - lenient enough to keep most genuine epochs
#   - strict enough to remove clear artifacts
REJECT_THRESHOLD = dict(eeg=150e-6)   # 150 µV

epochs_clean = epochs.copy().drop_bad(reject=REJECT_THRESHOLD)

n_dropped = len(epochs) - len(epochs_clean)
pct_dropped = 100 * n_dropped / len(epochs)
print(f"  Threshold              : {REJECT_THRESHOLD['eeg']*1e6:.0f} µV")
print(f"  Epochs before          : {len(epochs)}")
print(f"  Epochs after           : {len(epochs_clean)}")
print(f"  Dropped                : {n_dropped} ({pct_dropped:.1f}%)")

if pct_dropped > 30:
    print("\n  WARNING: >30% of epochs rejected.")
    print("     Consider raising the threshold if data quality is expected to be good,")
    print("     or inspect raw data for systematic noise.")

# 7. Save
print("\n" + "=" * 60)
print("STEP 6: Saving")
print("=" * 60)

X = epochs_clean.get_data()   # shape: (n_epochs, n_channels, n_times)
y = epochs_clean.events[:, -1]

np.save("X.npy", X)
np.save("y.npy", y)

print(f"  X.npy saved   shape: {X.shape}  (epochs × channels × times)")
print(f"  y.npy saved   shape: {y.shape}  labels: {np.unique(y)}")
print("\nPreprocessing complete.")