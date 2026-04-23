
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
N_PERMS    = 1000   # number of permutations (1000 is standard)
ALPHA      = 0.05   # significance threshold

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

#Load EEG 
print("Loading EEG...")
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

eeg_avg = np.mean(eegs, axis=0)
n_images, n_channels, n_times = eeg_avg.shape
print(f"  EEG shape: {eeg_avg.shape}")
TIMES_MS   = np.linspace(-200, 800, n_times)
#Load model RDMs 
resnet_rdm = np.load(os.path.join(RDM_DIR, "rdm_layer2.npy"))
clip_rdm   = np.load(os.path.join(RDM_DIR, "rdm_clip.npy"))

#Compute observed RSA curves 
print("Computing observed RSA curves...")
rsa_resnet = np.zeros(n_times)
rsa_clip   = np.zeros(n_times)

for t in range(n_times):
    brain_rdm_t   = compute_rdm(eeg_avg[:, :, t])
    rsa_resnet[t] = rdm_corr(resnet_rdm, brain_rdm_t)
    rsa_clip[t]   = rdm_corr(clip_rdm,   brain_rdm_t)

rsa_resnet_s = uniform_filter1d(rsa_resnet, size=5)
rsa_clip_s   = uniform_filter1d(rsa_clip,   size=5)

#Permutation test 
# Randomly shuffle image labels in the brain RDM and recompute RSA.
# This gives us a null distribution of r values at each timepoint.
print(f"\nRunning permutation test ({N_PERMS} permutations)...")
print("This takes a few minutes - please wait...")

null_resnet = np.zeros((N_PERMS, n_times))
null_clip   = np.zeros((N_PERMS, n_times))

rng = np.random.default_rng(42)

# Precompute all brain RDMs once (big speedup)
print("Precomputing brain RDMs...")
brain_rdms = [compute_rdm(eeg_avg[:, :, t]) for t in range(n_times)]

for p in range(N_PERMS):
    perm_idx = rng.permutation(n_images)
    for t in range(n_times):
        brain_rdm_perm   = brain_rdms[t][np.ix_(perm_idx, perm_idx)]
        null_resnet[p,t] = rdm_corr(resnet_rdm, brain_rdm_perm)
        null_clip[p,t]   = rdm_corr(clip_rdm,   brain_rdm_perm)

    if p % 200 == 0:
        print(f"  Permutation {p+1}/{N_PERMS}")

print("  Permutation test done!")

#Find significant timepoints (p < 0.05, one-tailed) 
#p-value = proportion of null distribution >= observed value
p_resnet = np.mean(null_resnet >= rsa_resnet[np.newaxis, :], axis=0)
p_clip   = np.mean(null_clip   >= rsa_clip[np.newaxis, :],   axis=0)

sig_resnet = p_resnet < ALPHA
sig_clip   = p_clip   < ALPHA

#Cluster correction 
# Only keep clusters of 3+ consecutive significant timepoints
def cluster_correct(sig_array, min_cluster=8):
    corrected = np.zeros_like(sig_array)
    in_cluster = False
    start = 0
    for i in range(len(sig_array)):
        if sig_array[i] and not in_cluster:
            in_cluster = True
            start = i
        elif not sig_array[i] and in_cluster:
            if i - start >= min_cluster:
                corrected[start:i] = True
            in_cluster = False
    if in_cluster and len(sig_array) - start >= min_cluster:
        corrected[start:] = True
    return corrected.astype(bool)

sig_resnet_c = cluster_correct(sig_resnet)
sig_clip_c   = cluster_correct(sig_clip)

print(f"\nSignificant timepoints (cluster-corrected):")
print(f"  ResNet50: {sig_resnet_c.sum()} timepoints")
print(f"  CLIP:     {sig_clip_c.sum()} timepoints")

#Load noise ceiling 
print("\nRecomputing noise ceiling...")
n_subs = len(eegs)
nc = np.zeros(n_times)
for t in range(n_times):
    corrs = []
    for i in range(n_subs):
        rdm_i    = compute_rdm(eegs[i][:, :, t])
        others   = [compute_rdm(eegs[j][:, :, t]) for j in range(n_subs) if j != i]
        rdm_rest = np.mean(others, axis=0)
        corrs.append(rdm_corr(rdm_i, rdm_rest))
    nc[t] = np.mean(corrs)
nc_s = uniform_filter1d(nc, size=5)

#Find peaks post-stimulus 
onset = 20
resnet_peak_idx = onset + int(np.argmax(rsa_resnet_s[onset:]))
clip_peak_idx   = onset + int(np.argmax(rsa_clip_s[onset:]))
resnet_peak_ms  = TIMES_MS[resnet_peak_idx]
clip_peak_ms    = TIMES_MS[clip_peak_idx]
resnet_peak_r   = rsa_resnet_s[resnet_peak_idx]
clip_peak_r     = rsa_clip_s[clip_peak_idx]

#Final Figure 
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("#FAFAFA")

# Pre-stimulus shading
ax.axvspan(-200, 0, alpha=0.06, color="gray")

# Noise ceiling
ax.fill_between(TIMES_MS, 0, nc_s,
                alpha=0.13, color="gray", label="Noise ceiling (lower bound)")
ax.plot(TIMES_MS, nc_s, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)

# Significance shading under curves
def shade_significant(ax, times, sig_array, color, alpha=0.18):
    """Draw one solid rectangle per contiguous significant cluster."""
    in_cluster = False
    start_t = None
    for i in range(len(sig_array)):
        if times[i] < 0:
            continue
        if sig_array[i] and not in_cluster:
            in_cluster = True
            start_t = times[i]
        elif not sig_array[i] and in_cluster:
            ax.axvspan(start_t, times[i], alpha=alpha, color=color, zorder=1)
            in_cluster = False
    if in_cluster:
        ax.axvspan(start_t, times[-1], alpha=alpha, color=color, zorder=1)

shade_significant(ax, TIMES_MS, sig_resnet_c, "#2ca02c", alpha=0.12)
shade_significant(ax, TIMES_MS, sig_clip_c,   "#d62728", alpha=0.12)

# Model curves
ax.plot(TIMES_MS, rsa_resnet_s, color="#2ca02c", linewidth=2.8,
        label="ResNet50 Layer 2 (vision-only)")
ax.plot(TIMES_MS, rsa_clip_s,   color="#d62728", linewidth=2.8,
        label="CLIP ViT-B/32 (vision + language)")

# Peak markers
ax.scatter([resnet_peak_ms], [resnet_peak_r], color="#2ca02c",
           s=90, zorder=10, edgecolors="black", linewidths=0.8)
ax.scatter([clip_peak_ms],   [clip_peak_r],   color="#d62728",
           s=90, zorder=10, edgecolors="black", linewidths=0.8)

ax.annotate(f"ResNet50\n{resnet_peak_ms:.0f}ms, r={resnet_peak_r:.3f}",
            xy=(resnet_peak_ms, resnet_peak_r),
            xytext=(resnet_peak_ms + 70, resnet_peak_r + 0.02),
            fontsize=9, color="#2ca02c", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5))

ax.annotate(f"CLIP\n{clip_peak_ms:.0f}ms, r={clip_peak_r:.3f}",
            xy=(clip_peak_ms, clip_peak_r),
            xytext=(clip_peak_ms - 170, clip_peak_r + 0.03),
            fontsize=9, color="#d62728", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5))

# Reference lines
ax.axvline(0, color="black", linewidth=1.5, linestyle="--", zorder=5)
ax.axhline(0, color="gray",  linewidth=0.8, linestyle=":")

ax.text(-190, 0.01, "Pre-stimulus\nbaseline", fontsize=8, color="gray")
ax.text(5, -0.04, "Shading = p<0.05\n(cluster-corrected)", fontsize=7.5, color="gray")

ax.set_xlabel("Time relative to stimulus onset (ms)", fontsize=12)
ax.set_ylabel("Spearman r  (Model - Brain RSA)", fontsize=12)
ax.set_title("ResNet50 vs. CLIP: Brain Alignment Over Time\n"
             "(3 participants, THINGS-EEG2, cluster-corrected p<0.05)",
             fontsize=12, fontweight="bold", pad=12)
ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(-200, 800)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "figure6_final.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved Figure 6 -> {path}")
