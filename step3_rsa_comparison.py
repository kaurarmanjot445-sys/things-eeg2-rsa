
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os

RDM_DIR    = "rdms/"
OUTPUT_DIR = "figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LAYER_NAMES  = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
LAYER_LABELS = ["Layer 1\n(shallow)", "Layer 2", "Layer 3",
                "Layer 4", "AvgPool\n(deep)"]

#Load RDMs 
print("Loading RDMs...")
dnn_rdms = {}
for layer in LAYER_NAMES:
    dnn_rdms[layer] = np.load(os.path.join(RDM_DIR, f"rdm_{layer}.npy"))

brain_rdm = np.load(os.path.join(RDM_DIR, "rdm_brain_eeg.npy"))
print("  All RDMs loaded OK")

#RSA: Spearman correlation 
def rdm_correlation(rdm_a, rdm_b):
    idx = np.triu_indices_from(rdm_a, k=1)
    r, p = spearmanr(rdm_a[idx], rdm_b[idx])
    return r, p

print("\nRSA Results:")
print("-" * 45)
rsa_scores = []
p_values   = []
for layer in LAYER_NAMES:
    r, p = rdm_correlation(dnn_rdms[layer], brain_rdm)
    rsa_scores.append(r)
    p_values.append(p)
    print(f"  {layer:10s}: r = {r:.4f}  (p = {p:.2e})")

best_idx   = int(np.argmax(rsa_scores))
best_layer = LAYER_NAMES[best_idx]
print(f"\n  Best layer: {best_layer}  (r = {rsa_scores[best_idx]:.4f})")


#FIGURE 1 - RDM Heatmaps
N_SHOW = 200  

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

all_rdms   = [dnn_rdms[l] for l in LAYER_NAMES] + [brain_rdm]
all_labels = LAYER_LABELS + ["EEG Brain"]

for i, (rdm, label) in enumerate(zip(all_rdms, all_labels)):
    ax  = axes[i]
    sub = rdm[:N_SHOW, :N_SHOW]
    im  = ax.imshow(sub, cmap="viridis", aspect="auto",
                    vmin=0, vmax=np.percentile(rdm, 95))
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel("Image index")
    ax.set_ylabel("Image index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Representational Dissimilarity Matrices\nResNet50 Layers vs. EEG Brain",
             fontsize=14, fontweight="bold")
plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, "figure1_rdm_heatmaps.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved Figure 1 -> {path1}")


#FIGURE 2 - RSA Profile(KEY result)
fig, ax = plt.subplots(figsize=(7, 5))

colors = ["#E05C2A" if i == best_idx else "#4A90D9"
          for i in range(len(LAYER_NAMES))]

bars = ax.bar(LAYER_LABELS, rsa_scores, color=colors,
              edgecolor="black", linewidth=0.7, width=0.55)

for bar, r, p in zip(bars, rsa_scores, p_values):
    sig = "*" if p < 0.05 else "n.s."
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"r={r:.3f}{sig}",
            ha="center", va="bottom", fontsize=9)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_ylabel("Spearman r (DNN-Brain RSA)", fontsize=11)
ax.set_xlabel("ResNet50 Layer (shallow to deep)", fontsize=11)
ax.set_title("Which DNN layer best predicts\nbrain representations?",
             fontsize=12, fontweight="bold")
ax.set_ylim(min(0, min(rsa_scores)) - 0.05, max(rsa_scores) + 0.07)
ax.spines[["top", "right"]].set_visible(False)

ax.annotate("Best match\nto brain",
            xy=(best_idx, rsa_scores[best_idx]),
            xytext=(best_idx + 0.6, rsa_scores[best_idx] + 0.03),
            fontsize=9, color="#E05C2A",
            arrowprops=dict(arrowstyle="->", color="#E05C2A"))

plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, "figure2_rsa_profile.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved Figure 2 -> {path2}")

print("\nAll done! Check your figures/ folder for your 2 paper figures.")
print(f"Key finding: {best_layer} of ResNet50 best predicts EEG brain responses")
print(f"Spearman r = {rsa_scores[best_idx]:.4f}")