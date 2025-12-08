import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
import io
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- MODEL DEFINITIONS (From api.py) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class VGG16UNET_preent(nn.Module):
    def __init__(self, n_channels=3, n_classes=8):
        super(VGG16UNET_preent, self).__init__()
        # Load VGG16 with strict=False or prevent download if possible.
        # However, since the weights rely on the feature extractor structure,
        # we stick to the original initialization but warn about memory.
        # Optimization: We could use weights=None if our pth file is full,
        # but to be safe and match original logic, we use pretrained=True
        # because the user's pth might be state_dict() OF THIS CLASS,
        # which means it expects this architecture.
        # We will cache the model loading so this happens only once.
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Sequential(*list(vgg16.features.children())[:4])
        self.down1 = nn.Sequential(*list(vgg16.features.children())[4:9])
        self.down2 = nn.Sequential(*list(vgg16.features.children())[9:16])
        self.down3 = nn.Sequential(*list(vgg16.features.children())[16:23])

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)

# --- UTILS ---
color_palette = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128]
]

class_mapping = {
    0: 0, 1: 1, 4: 2, 5: 3, 2: 4, 3: 5, 6: 6, 7: 7
}

class_names = [
    "Void", "Flat", "Construction", "Object",
    "Nature", "Sky", "Human", "Vehicle"
]

def calculate_iou(pred_mask, true_mask, num_classes):
    iou_list = []
    for cls in range(num_classes):
        pred_inds = pred_mask == cls
        target_inds = true_mask == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        iou = intersection / (union + 1e-8)
        iou_list.append(iou)
    return np.mean(iou_list)

def calculate_dice_coefficient(outputs, targets, num_classes):
    dice_list = []
    outputs = torch.argmax(outputs, dim=1)
    for cls in range(num_classes):
        pred_inds = (outputs == cls)
        target_inds = (targets == cls)
        intersection = torch.logical_and(pred_inds, target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item()
        dice = (2.0 * intersection) / (union + 1e-8)
        dice_list.append(dice)
    return sum(dice_list) / len(dice_list)

def remap_mask_to_classes(mask_array):
    gray_to_class = {
        0: 0, 15: 1, 38: 2, 53: 3, 75: 4, 90: 5, 113: 6, 128: 7
    }
    remapped_mask = np.zeros_like(mask_array, dtype=np.uint8)
    for gray_value in np.unique(mask_array):
        if gray_value in gray_to_class:
            remapped_mask[mask_array == gray_value] = gray_to_class[gray_value]
    return remapped_mask

# --- APP LOGIC ---

st.set_page_config(
    page_title="Segmentation Embarquée",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Model with caching to save resources
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = Path(__file__).parent / 'vgg16unet_pt.pth'
    
    try:
        model = VGG16UNET_preent(n_channels=3, n_classes=8)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Erreur de chargement du modèle: {e}")
        return None, None

model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
])

st.title("Visualisation de la Segmentation d'une image (Cloud Optimized)")
st.caption("Version optimisée pour le déploiement Cloud sans API séparée")

alpha = st.sidebar.slider("Transparence du masque", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])
uploaded_file_mask = st.file_uploader("Choisir le masque de l'image (Optionnel)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Processing
    image = Image.open(uploaded_file).convert("RGB")
    original_size = image.size
    
    # Inference
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    
    predicted_mask_raw = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Remap
    remapped_mask = np.zeros_like(predicted_mask_raw, dtype=np.uint8)
    for p_idx, r_idx in class_mapping.items():
        remapped_mask[predicted_mask_raw == p_idx] = r_idx
        
    remapped_mask_resized = cv2.resize(
        remapped_mask,
        (original_size[0], original_size[1]),
        interpolation=cv2.INTER_NEAREST
    )
    
    # Colors
    colored_mask = np.zeros((remapped_mask_resized.shape[0], remapped_mask_resized.shape[1], 3), dtype=np.uint8)
    for class_idx in range(len(color_palette)):
        colored_mask[remapped_mask_resized == class_idx] = color_palette[class_idx]

    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image Originale")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Prédiction")
        st.image(colored_mask, caption="Masque Prédit", use_container_width=True)
        
    st.markdown("---")
    
    # Stats Calculation
    if uploaded_file_mask:
        true_mask_img = Image.open(uploaded_file_mask)
        if true_mask_img.mode not in ["L", "P", "1"]:
            true_mask_img = true_mask_img.convert("L")
        true_mask_resized = true_mask_img.resize((512, 256), Image.NEAREST)
        true_mask = np.array(true_mask_resized)
        true_mask_remapped = remap_mask_to_classes(true_mask)
        true_mask_tensor = torch.tensor(true_mask_remapped).unsqueeze(0).to(device)
        
        iou = calculate_iou(predicted_mask_raw, true_mask_remapped, 8)
        dice = calculate_dice_coefficient(output, true_mask_tensor, 8)
        
        st.success(f"**IoU**: {iou:.4f} | **Dice**: {dice:.4f}")
        
    # Overlay
    image_np = np.array(image)
    overlay = cv2.addWeighted(image_np, 1 - alpha, colored_mask, alpha, 0)
    st.subheader("Superposition")
    st.image(overlay, use_container_width=True)
    
    # Class Dist
    pixel_counts = {}
    total_pixels = remapped_mask_resized.size
    for class_idx, class_name in enumerate(class_names):
        count = np.sum(remapped_mask_resized == class_idx)
        percentage = (count / total_pixels) * 100
        if percentage > 0.0:
            pixel_counts[class_name] = percentage

    st.subheader("Répartition")
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(pixel_counts.keys(), pixel_counts.values())
    ax.bar_label(bars, fmt='%.1f%%')
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif model is None:
    st.error("Le modèle n'a pas pu être chargé.")
