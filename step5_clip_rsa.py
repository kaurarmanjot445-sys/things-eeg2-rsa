
import torch
import open_clip
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

IMAGE_DIR  = "test_images/"
RDM_DIR    = "rdms/"
OUTPUT_DIR = "figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EEG_PATHS = [
    "sub-01/preprocessed_eeg_test.npy",
    "sub-02/preprocessed_eeg_test.npy",
    "sub-03/preprocessed_eeg_test.npy",
]
TIMES_MS = np.linspace(-200, 800, 100)

#Load CLIP 
print("Loading CLIP model (ViT-B/32)...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model.eval()
print("  CLIP loaded OK")

#Find images 
image_files = []
for subfolder in sorted(os.listdir(IMAGE_DIR)):
    subfolder_path = os.path.join(IMAGE_DIR, subfolder)
    if os.path.isdir(subfolder_path):
        for fname in sorted(os.listdir(subfolder_path)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(subfolder_path, fname))
                break

print(f"  Found {len(image_files)} images")

#Extract CLIP image embeddings 
print("\nExtracting CLIP activations...")
clip_features = []

with torch.no_grad():
    for img_path in tqdm(image_files, desc="CLIP"):
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0)
        features = model.encode_image(tensor)
        clip_features.append(features.cpu().numpy())

clip_features = np.concatenate(clip_features, axis=0)
print(f"  CLIP features shape: {clip_features.shape}")

#Save CLIP activations
np.save(os.path.join(RDM_DIR, "activations_clip.npy"), clip_features)

#Compute CLIP RDM 
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

print("\nComputing CLIP RDM...")
clip_rdm = compute_rdm(clip_features)
np.save(os.path.join(RDM_DIR, "rdm_clip.npy"), clip_rdm)
print(f"  CLIP RDM shape: {clip_rdm.shape}")

resnet_rdm = np.load(os.path.join(RDM_DIR, "rdm_layer2.npy"))
print(f"  ResNet Layer 2 RDM shape: {resnet_rdm.shape}")

#Load + average EEG 
print("\nLoading EEG data...")
eegs = []
for path in EEG_PATHS:
    e = np.load(path, allow_pickle=True)
    if e.ndim == 0:
        e = e.item()
        e = list(e.values())[0] if isinstance(e, dict) else e
        e = np.array(e)
    eegs.append(e)
eeg = np.mean(eegs, axis=0)
if eeg.ndim == 4:
    eeg_avg = eeg.mean(axis=1)
else:
    eeg_avg = eeg
n_images, n_channels, n_times = eeg_avg.shape
print(f"  EEG shape: {eeg_avg.shape}")

#Time-resolved RSA for both models 
print("\nRunning time-resolved RSA for ResNet50 vs CLIP...")
rsa_resnet = np.zeros(n_times)
rsa_clip   = np.zeros(n_times)

for t in range(n_times):
    eeg_t       = eeg_avg[:, :, t]
    brain_rdm_t = compute_rdm(eeg_t)
    rsa_resnet[t] = rdm_corr(resnet_rdm, brain_rdm_t)
    rsa_clip[t]   = rdm_corr(clip_rdm,   brain_rdm_t)
    if t % 20 == 0:
        print(f"  Timepoint {t+1}/{n_times}")

# Smooth
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

print(f"\nResNet50 Layer 2 peak: {resnet_peak_ms:.0f}ms  r={resnet_peak_r:.3f}")
print(f"CLIP ViT-B/32 peak:    {clip_peak_ms:.0f}ms  r={clip_peak_r:.3f}")

#Figure 4: ResNet vs CLIP time-resolved RSA 
fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#FAFAFA")

# Shade pre-stimulus
ax.axvspan(-200, 0, alpha=0.06, color="gray")

# Plot both models
ax.plot(TIMES_MS, rsa_resnet_s, color="#2ca02c", linewidth=2.8,
        label="ResNet50 Layer 2 (vision-only)")
ax.plot(TIMES_MS, rsa_clip_s, color="#d62728", linewidth=2.8,
        label="CLIP ViT-B/32 (vision + language)")

# Mark peaks
ax.scatter([resnet_peak_ms], [resnet_peak_r], color="#2ca02c",
           s=80, zorder=10, edgecolors="black", linewidths=0.8)
ax.scatter([clip_peak_ms], [clip_peak_r], color="#d62728",
           s=80, zorder=10, edgecolors="black", linewidths=0.8)

ax.annotate(f"ResNet50\n{resnet_peak_ms:.0f}ms, r={resnet_peak_r:.3f}",
            xy=(resnet_peak_ms, resnet_peak_r),
            xytext=(resnet_peak_ms + 80, resnet_peak_r + 0.02),
            fontsize=9, color="#2ca02c", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5))

ax.annotate(f"CLIP\n{clip_peak_ms:.0f}ms, r={clip_peak_r:.3f}",
            xy=(clip_peak_ms, clip_peak_r),
            xytext=(clip_peak_ms - 180, clip_peak_r + 0.02),
            fontsize=9, color="#d62728", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5))

ax.axvline(0, color="black", linewidth=1.5, linestyle="--", zorder=5)
ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

ax.text(-190, ax.get_ylim()[1] * 0.88, "Pre-stimulus\nbaseline",
        fontsize=8, color="gray", va="top")

ax.set_xlabel("Time relative to stimulus onset (ms)", fontsize=12)
ax.set_ylabel("Spearman r  (Model - Brain RSA)", fontsize=12)
ax.set_title("Vision-Only vs. Vision-Language Model:\nWhich Aligns Better with Human EEG?",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(-200, 800)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "figure4_resnet_vs_clip.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved Figure 4 -> {path}")