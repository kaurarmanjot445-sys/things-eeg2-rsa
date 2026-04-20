
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os

ACTIVATION_DIR = "activations/"
EEG_PATH       = "sub-01/preprocessed_eeg_test.npy"
OUTPUT_DIR     = "rdms/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LAYER_NAMES = ["layer1", "layer2", "layer3", "layer4", "avgpool"]

def compute_rdm(activation_matrix):
    X = activation_matrix - activation_matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    X = X / norms
    rdm = squareform(pdist(X, metric="correlation"))
    return rdm

print("Computing DNN RDMs...")
dnn_rdms = {}
for layer in LAYER_NAMES:
    path = os.path.join(ACTIVATION_DIR, f"{layer}.npy")
    acts = np.load(path)
    rdm = compute_rdm(acts)
    dnn_rdms[layer] = rdm
    np.save(os.path.join(OUTPUT_DIR, f"rdm_{layer}.npy"), rdm)
    print(f"  {layer}: RDM shape {rdm.shape}")

print("\nLoading EEG data...")
eeg = np.load(EEG_PATH, allow_pickle=True)
if eeg.ndim == 0:
    eeg = eeg.item()
    eeg = list(eeg.values())[0] if isinstance(eeg, dict) else eeg
    eeg = np.array(eeg)
print(f"  Raw EEG shape: {eeg.shape}")
if eeg.ndim == 4:
    eeg_avg = eeg.mean(axis=1)
    n_images = eeg_avg.shape[0]
    eeg_flat = eeg_avg.reshape(n_images, -1)
elif eeg.ndim == 3:
    n_images = eeg.shape[0]
    eeg_flat = eeg.reshape(n_images, -1)
else:
    eeg_flat = eeg

print(f"  EEG matrix for RDM: {eeg_flat.shape}")

brain_rdm = compute_rdm(eeg_flat)
print(f"  Brain RDM shape: {brain_rdm.shape}")

n_eeg = brain_rdm.shape[0]
n_dnn = dnn_rdms["layer1"].shape[0]

if n_dnn != n_eeg:
    print(f"\n  Trimming DNN RDMs from {n_dnn} to {n_eeg} images to match EEG.")
    for layer in LAYER_NAMES:
        dnn_rdms[layer] = dnn_rdms[layer][:n_eeg, :n_eeg]
        np.save(os.path.join(OUTPUT_DIR, f"rdm_{layer}.npy"), dnn_rdms[layer])
        print(f"  {layer}: trimmed to {dnn_rdms[layer].shape}")

np.save(os.path.join(OUTPUT_DIR, "rdm_brain_eeg.npy"), brain_rdm)
print("\nSaved brain RDM.")
print("Done! Now run step3_rsa_comparison.py")