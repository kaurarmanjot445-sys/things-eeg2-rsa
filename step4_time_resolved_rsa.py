
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

EEG_PATHS = [
    "sub-01/preprocessed_eeg_test.npy",
    "sub-02/preprocessed_eeg_test.npy",
    "sub-03/preprocessed_eeg_test.npy",
]
RDM_DIR    = "rdms/"
OUTPUT_DIR = "figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LAYER_NAMES  = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
LAYER_LABELS = ["Layer 1 (shallow)", "Layer 2", "Layer 3", "Layer 4", "AvgPool (deep)"]
COLORS       = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

#Load + average EEG across participants 
print("Loading EEG data...")
eegs = []
for path in EEG_PATHS:
    e = np.load(path, allow_pickle=True)
    if e.ndim == 0:
        e = e.item()
        e = list(e.values())[0] if isinstance(e, dict) else e
        e = np.array(e)
    eegs.append(e)
eeg = np.mean(eegs, axis=0)
print(f"  Averaged {len(eegs)} participants")
print(f"  EEG shape: {eeg.shape}")

if eeg.ndim == 4:
    eeg_avg = eeg.mean(axis=1)
elif eeg.ndim == 3:
    eeg_avg = eeg

n_images, n_channels, n_times = eeg_avg.shape
TIMES_MS     = np.linspace(-200, 800, n_times)
#Load DNN RDMs 
print("Loading DNN RDMs...")
dnn_rdms = {}
for layer in LAYER_NAMES:
    dnn_rdms[layer] = np.load(os.path.join(RDM_DIR, f"rdm_{layer}.npy"))

#Helper functions 
def compute_rdm(X):
    X = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    X = X / norms
    return squareform(pdist(X, metric="correlation"))

def rdm_corr(rdm_a, rdm_b):
    idx = np.triu_indices_from(rdm_a, k=1)
    r, _ = spearmanr(rdm_a[idx], rdm_b[idx])
    return r

#Time-resolved RSA 
print(f"\nRunning time-resolved RSA across {n_times} timepoints...")
rsa_time = np.zeros((len(LAYER_NAMES), n_times))

for t in range(n_times):
    eeg_t = eeg_avg[:, :, t]
    brain_rdm_t = compute_rdm(eeg_t)
    for li, layer in enumerate(LAYER_NAMES):
        rsa_time[li, t] = rdm_corr(dnn_rdms[layer], brain_rdm_t)
    if t % 20 == 0:
        print(f"  Timepoint {t+1}/{n_times}")

rsa_smooth = uniform_filter1d(rsa_time, size=5, axis=1)

np.save(os.path.join(OUTPUT_DIR, "rsa_time_resolved.npy"), rsa_smooth)

#Find best layer and its peak ─
#Only look after stimulus onset (t=0 = index 20)
onset_idx = 20
peak_per_layer = [rsa_smooth[li, onset_idx:].max() for li in range(len(LAYER_NAMES))]
best_layer_idx = int(np.argmax(peak_per_layer))
best_layer_name = LAYER_NAMES[best_layer_idx]
peak_t_idx = onset_idx + int(np.argmax(rsa_smooth[best_layer_idx, onset_idx:]))
peak_t_ms  = TIMES_MS[peak_t_idx]
peak_r     = rsa_smooth[best_layer_idx, peak_t_idx]

print(f"\nBest layer post-stimulus: {best_layer_name}")
print(f"Peak at {peak_t_ms:.0f}ms, r={peak_r:.3f}")

#Plot 
fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#FAFAFA")

# Shade pre-stimulus baseline
ax.axvspan(-200, 0, alpha=0.06, color="gray", zorder=0)

#Plot each layer
for li, (layer, label, color) in enumerate(zip(LAYER_NAMES, LAYER_LABELS, COLORS)):
    lw = 3.0 if li == best_layer_idx else 1.8
    alpha = 1.0 if li == best_layer_idx else 0.7
    ax.plot(TIMES_MS, rsa_smooth[li], label=label, color=color,
            linewidth=lw, alpha=alpha)

#Stimulus onset line
ax.axvline(0, color="black", linewidth=1.5, linestyle="--", zorder=5)
ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", zorder=5)

# Mark peak
ax.scatter([peak_t_ms], [peak_r], color=COLORS[best_layer_idx],
           s=80, zorder=10, edgecolors="black", linewidths=0.8)
ax.annotate(f"{LAYER_LABELS[best_layer_idx]}\npeak {peak_t_ms:.0f}ms\nr={peak_r:.3f}",
            xy=(peak_t_ms, peak_r),
            xytext=(peak_t_ms + 90, peak_r + 0.025),
            fontsize=9.5, color=COLORS[best_layer_idx], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS[best_layer_idx], lw=1.5))

#Labels
ax.text(-190, ax.get_ylim()[1] * 0.92, "Pre-stimulus\nbaseline",
        fontsize=8, color="gray", va="top")
ax.text(10, ax.get_ylim()[1] * 0.92, "Post-stimulus",
        fontsize=8, color="gray", va="top")

ax.set_xlabel("Time relative to stimulus onset (ms)", fontsize=12)
ax.set_ylabel("Spearman r  (DNN - Brain RSA)", fontsize=12)
ax.set_title("Time-Resolved RSA: When Do ResNet50 Layers Align with Human EEG?\n"
             "(averaged across 3 participants, THINGS-EEG2)",
             fontsize=12, fontweight="bold", pad=12)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(-200, 800)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "figure3_time_resolved_rsa.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved Figure 3 -> {path}")
print("Done!")