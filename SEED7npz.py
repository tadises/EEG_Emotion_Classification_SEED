import os, glob
import mne, numpy as np

def extract_label(fif_path):
    basename = os.path.basename(fif_path)
    emo = basename.split('_')[0]
    mapping = {
        'Disgust': 0, 'Fear': 1, 'Sad': 2,
        'Neutral': 3, 'Happy': 4,
        'Anger': 5,  'Surprise': 6
    }
    return mapping[emo]

raw_dir = r'K:\Download\SEED-VII\SEED-VII\epochs_data\fif'
out_dir = r'K:\Download\SEED-VII\SEED-VII\epochs_data\npz'
os.makedirs(out_dir, exist_ok=True)

window = 200
step = window  # 不重叠；要重叠就改成 window-overlap

for fif_path in glob.glob(os.path.join(raw_dir, '*-epo.fif')):
    epochs = mne.read_epochs(fif_path, preload=True)
    arr = epochs.get_data()            # (n_epochs, 62, T)
    label = extract_label(fif_path)
    base = os.path.splitext(os.path.basename(fif_path))[0]
    for ei in range(arr.shape[0]):
        sig = arr[ei]                  # (62, T)
        T = sig.shape[1]
        for start in range(0, T-window+1, step):
            seg = sig[:, start:start+window]  # (62,200)
            np.savez(
                os.path.join(out_dir, f"{base}_e{ei}_s{start}.npz"),
                data=seg, label=label
            )
