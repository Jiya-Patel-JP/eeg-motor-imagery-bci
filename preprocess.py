import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

# Load data
# Subject 1, runs 6/10/14 = imagined fist vs feet movement
print("Downloading/loading data...")
raw_files = [read_raw_edf(f, preload=True, stim_channel='auto')
             for f in eegbci.load_data(1, [6, 10, 14])]
raw = concatenate_raws(raw_files)
print("Raw data loaded:", raw)

# Clean up channel names
eegbci.standardize(raw)
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)

# Bandpass filter: 8–30 Hz (alpha + beta)
print("Filtering...")
raw.filter(8., 30., fir_design='firwin', skip_by_annotation='edge')

# Extract events & epoch
events, event_id = mne.events_from_annotations(raw)
print("Event types found:", event_id)

# T1 = imagined fist, T2 = imagined feet
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)
epochs = mne.Epochs(raw, events, event_id=dict(fist=2, feet=3),
                    tmin=0., tmax=4., proj=True,
                    picks=picks, baseline=None, preload=True)

print("Epochs created:", epochs)
print("Shape:", epochs.get_data().shape)  # (n_epochs, n_channels, n_times)

# Save for next step
np.save('X.npy', epochs.get_data())
np.save('y.npy', epochs.events[:, -1])
print("Saved X.npy and y.npy — preprocessing done!")
