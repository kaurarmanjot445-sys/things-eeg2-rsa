
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import os

EEG_PATHS = [
    "sub-01/preprocessed_eeg_test.npy",
    "sub-02/preprocessed_eeg_test.npy",
    "sub-03/preprocessed_eeg_test.npy",
]
RDM_DIR    = "rdms/"
OUTPUT_DIR = "figures/"

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

#Load all 3 participants 
print("Loading EEG per participant...")
eegs = []
for path in EEG_PATHS:
    e = np.load(path, allow_pickle=True)
    if e.ndim == 0:
        e = e.item()
        e = list(e.values())[0] if isinstance(e, dict) else e
        e = np.array(e)
    if e.ndim == 4:
        e = e.mean(axis=1)
    eegs.append(e)
    print(f"  Loaded: {e.shape}")

n_subs  = len(eegs)
n_times = eegs[0].shape[2]
TIMES_MS   = np.linspace(-200, 800, n_times)

#Noise ceiling: lower bound only (most honest with 3 participants) 
#For each participant, correlate their RDM with the average of the OTHER two.
#This is the standard lower-bound noise ceiling.
print("\nComputing noise ceiling (lower bound)...")
nc = np.zeros(n_times)

for t in range(n_times):
    corrs = []
    for i in range(n_subs):
        #RDM for this participant
        rdm_i = compute_rdm(eegs[i][:, :, t])
        #Average RDM of the other participants
        others   = [compute_rdm(eegs[j][:, :, t]) for j in range(n_subs) if j != i]
        rdm_rest = np.mean(others, axis=0)
        corrs.append(rdm_corr(rdm_i, rdm_rest))
    nc[t] = np.mean(corrs)
    if t % 20 == 0:
        print(f"  Timepoint {t+1}/{n_times}  nc={nc[t]:.3f}")

nc_s = uniform_filter1d(nc, size=5)

#Sanity check - pre-stimulus should be near 0
pre_stim_mean = nc_s[:20].mean()
post_stim_max = nc_s[20:].max()
print(f"\nPre-stimulus NC mean:  {pre_stim_mean:.3f}  (should be near 0)")
print(f"Post-stimulus NC max:  {post_stim_max:.3f}")

#Load model RSA curves 
resnet_rdm = np.load(os.path.join(RDM_DIR, "rdm_layer2.npy"))
clip_rdm   = np.load(os.path.join(RDM_DIR, "rdm_clip.npy"))

eeg_avg = np.mean(eegs, axis=0)
rsa_resnet = np.zeros(n_times)
rsa_clip   = np.zeros(n_times)

for t in range(n_times):
    brain_rdm_t    = compute_rdm(eeg_avg[:, :, t])
    rsa_resnet[t]  = rdm_corr(resnet_rdm, brain_rdm_t)
    rsa_clip[t]    = rdm_corr(clip_rdm,   brain_rdm_t)

rsa_resnet_s = uniform_filter1d(rsa_resnet, size=5)
rsa_clip_s   = uniform_filter1d(rsa_clip,   size=5)

#Find peaks post-stimulus 
onset = 20
resnet_peak_idx = onset + int(np.argmax(rsa_resnet_s[onset:]))
clip_peak_idx   = onset + int(np.argmax(rsa_clip_s[onset:]))
resnet_peak_ms  = TIMES_MS[resnet_peak_idx]
clip_peak_ms    = TIMES_MS[clip_peak_idx]
resnet_peak_r   = rsa_resnet_s[resnet_peak_idx]
clip_peak_r     = rsa_clip_s[clip_peak_idx]

print(f"\nResNet50 Layer 2: peak {resnet_peak_ms:.0f}ms, r={resnet_peak_r:.3f}")
print(f"CLIP ViT-B/32:    peak {clip_peak_ms:.0f}ms, r={clip_peak_r:.3f}")
print(f"Noise ceiling peak: {post_stim_max:.3f}")

#Figure 5: Final figure 
fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#FAFAFA")

ax.axvspan(-200, 0, alpha=0.06, color="gray")

# Noise ceiling as shaded region between 0 and nc (lower bound)
ax.fill_between(TIMES_MS, 0, nc_s,
                alpha=0.15, color="gray", label="Noise ceiling (lower bound)")
ax.plot(TIMES_MS, nc_s, color="gray", linewidth=1.2,
        linestyle="--", alpha=0.7)

# Model curves
ax.plot(TIMES_MS, rsa_resnet_s, color="#2ca02c", linewidth=2.8,
        label="ResNet50 Layer 2 (vision-only)")
ax.plot(TIMES_MS, rsa_clip_s,   color="#d62728", linewidth=2.8,
        label="CLIP ViT-B/32 (vision + language)")

# Mark peaks
ax.scatter([resnet_peak_ms], [resnet_peak_r], color="#2ca02c",
           s=80, zorder=10, edgecolors="black", linewidths=0.8)
ax.scatter([clip_peak_ms],   [clip_peak_r],   color="#d62728",
           s=80, zorder=10, edgecolors="black", linewidths=0.8)

ax.annotate(f"ResNet50\n{resnet_peak_ms:.0f}ms, r={resnet_peak_r:.3f}",
            xy=(resnet_peak_ms, resnet_peak_r),
            xytext=(resnet_peak_ms + 70, resnet_peak_r + 0.02),
            fontsize=9, color="#2ca02c", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5))

ax.annotate(f"CLIP\n{clip_peak_ms:.0f}ms, r={clip_peak_r:.3f}",
            xy=(clip_peak_ms, clip_peak_r),
            xytext=(clip_peak_ms - 160, clip_peak_r + 0.03),
            fontsize=9, color="#d62728", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5))

ax.axvline(0, color="black", linewidth=1.5, linestyle="--", zorder=5)
ax.axhline(0, color="gray",  linewidth=0.8, linestyle=":")

ax.text(-190, 0.02, "Pre-stimulus\nbaseline", fontsize=8, color="gray")

ax.set_xlabel("Time relative to stimulus onset (ms)", fontsize=12)
ax.set_ylabel("Spearman r  (Model - Brain RSA)", fontsize=12)
ax.set_title("ResNet50 vs. CLIP Brain Alignment with Noise Ceiling\n"
             "(3 participants, THINGS-EEG2, lower-bound noise ceiling)",
             fontsize=12, fontweight="bold", pad=12)
ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(-200, 800)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "figure5_with_noise_ceiling.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved Figure 5 -> {path}")
print("Done!")