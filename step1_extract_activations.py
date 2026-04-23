
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

#Config 
IMAGE_DIR  = "test_images/"   # folder with your THINGS stimulus images
OUTPUT_DIR = "activations/"     # where to save extracted activations
BATCH_SIZE = 32                 # speeds things up vs doing 1 image at a time
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Load pretrained ResNet50 
print("Loading ResNet50...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

#Define which layers to extract 
LAYERS = {
    "layer1": model.layer1,   # shallow - low-level edges
    "layer2": model.layer2,
    "layer3": model.layer3,
    "layer4": model.layer4,   # deep- high-level semantics
    "avgpool": model.avgpool, # global pooling
}

activations = {name: [] for name in LAYERS}

def make_hook(name):
    def hook(module, input, output):
        out = output.detach().cpu()
        if out.dim() == 4:   
            out = out.mean(dim=[2, 3])
        elif out.dim() == 3:
            out = out.squeeze(-1)
        activations[name].append(out.numpy())
    return hook

# Register hooks
handles = []
for name, layer in LAYERS.items():
    handles.append(layer.register_forward_hook(make_hook(name)))

#Image preprocessing 
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

#Find all images (handles upper and lowercase extensions) 
image_files = []
for subfolder in sorted(os.listdir(IMAGE_DIR)):
    subfolder_path = os.path.join(IMAGE_DIR, subfolder)
    if os.path.isdir(subfolder_path):
        for fname in sorted(os.listdir(subfolder_path)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(subfolder_path, fname))
                break

print(f"Found {len(image_files)} images.")

if len(image_files) == 0:
    print("ERROR: No images found in things_images/ folder!")
    print("Make sure your images are in the right place.")
    exit()

#Run images through model in batches 
def load_batch(file_list):
    tensors = []
    for img_path in file_list:
        try:
            img = Image.open(img_path).convert("RGB")
            tensors.append(preprocess(img))
        except Exception as e:
            print(f"  Skipping {img_path}: {e}")
    if len(tensors) == 0:
        return None
    return torch.stack(tensors)

print("Extracting activations (this will take a while on CPU)...")
with torch.no_grad():
    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Batches"):
        batch_files = image_files[i : i + BATCH_SIZE]
        batch_tensor = load_batch(batch_files)
        if batch_tensor is not None:
            model(batch_tensor)

# Remove hooks
for h in handles:
    h.remove()

#Save activations 
print("\nSaving activations...")
for name in LAYERS:
    arr = np.concatenate(activations[name], axis=0)  # (N_images, N_features)
    save_path = os.path.join(OUTPUT_DIR, f"{name}.npy")
    np.save(save_path, arr)
    print(f"  Saved {name}: shape {arr.shape} -> {save_path}")
print("\nDone! Now run step2_compute_rdms.py")