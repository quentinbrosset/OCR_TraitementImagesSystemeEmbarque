import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from PIL import Image
import numpy as np
import cv2
import io
from pathlib import Path

# Définir les composants du modèle (inchangé)
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

        # Ajuster les dimensions si nécessaire
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])

        # Concaténer
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
        # Utiliser un VGG16 pré-entraîné comme encodeur
        vgg16 = models.vgg16(pretrained=True)
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder (features de VGG16)
        self.inc = nn.Sequential(*list(vgg16.features.children())[:4])    # 64 canaux
        self.down1 = nn.Sequential(*list(vgg16.features.children())[4:9])  # 128 canaux
        self.down2 = nn.Sequential(*list(vgg16.features.children())[9:16]) # 256 canaux
        self.down3 = nn.Sequential(*list(vgg16.features.children())[16:23]) # 512 canaux

        # Decoder
        self.up1 = Up(512, 256)   # Concaténation des features de down2 et down3
        self.up2 = Up(256, 128)   # Concaténation des features de down1 et up1
        self.up3 = Up(128, 64)    # Concaténation des features de inc et up2

        # Couche de sortie
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # 64 canaux
        x2 = self.down1(x1)   # 128 canaux
        x3 = self.down2(x2)   # 256 canaux
        x4 = self.down3(x3)   # 512 canaux

        # Decoder avec skip connections
        x = self.up1(x4, x3)  # 256 canaux
        x = self.up2(x, x2)   # 128 canaux
        x = self.up3(x, x1)   # 64 canaux

        return self.outc(x)    # n_classes canaux

# Initialiser l'application FastAPI
app = FastAPI()

# Charger le modèle sauvegardé
model_path = Path(__file__).resolve().parent / 'vgg16unet_pt.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    # Initialiser le modèle
    model = VGG16UNET_preent(n_channels=3, n_classes=8)

    # Charger les poids du modèle
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Modèle chargé avec succès")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    model = None

# Transfo pour les images d'entrée
transform = transforms.Compose([
    transforms.Resize((256, 512)),   # Redimensionner comme dans votre code d'entraînement
    transforms.ToTensor(),
])

# Couleurs pour les 8 catégories
color_palette = [
                [0, 0, 0],        # Classe 0 - Void
                [128, 0, 0],      # Classe 1 - Flat
                [0, 128, 0],      # Classe 2 - Construction
                [128, 128, 0],    # Classe 3 - Object
                [0, 0, 128],      # Classe 4 - Nature
                [128, 0, 128],    # Classe 5 - Sky
                [0, 128, 128],    # Classe 6 - Human
                [128, 128, 128]   # Classe 7 - Vehicle
                    ]

class_mapping = {
    0: 0,  # Noir -> Void
    1: 1,  # Bleu foncé -> Flat
    4: 2,  # Rouge -> Nature
    5: 3,  # Cyan -> Object
    2: 4,  # Vert -> Construction
    3: 5,  # Magenta -> Sky
    6: 6,  # Jaune -> Human
    7: 7   # Gris clair -> Vehicle
}

# Nom des catégories
class_names = [
    "Void", "Flat", "Construction", "Object",
    "Nature", "Sky", "Human", "Vehicle"
]

# Fonction pour calculer l'IoU
def calculate_iou(pred_mask, true_mask, num_classes):
    iou_list = []

    # Calculer IoU pour chaque classe
    for cls in range(num_classes):
        pred_inds = pred_mask == cls
        target_inds = true_mask == cls

        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()

        iou = intersection / (union + 1e-8)
        iou_list.append(iou)

    return np.mean(iou_list)   # Retourne la moyenne des IoU

# Fonction pour calculer le coefficient de Dice - MODIFIÉE
def calculate_dice_coefficient(outputs, targets, num_classes):
    dice_list = []

    # Obtenir les prédictions
    outputs = torch.argmax(outputs, dim=1)

    # Calculer Dice pour chaque classe
    for cls in range(num_classes):
        pred_inds = (outputs == cls)
        target_inds = (targets == cls)
        
        intersection = torch.logical_and(pred_inds, target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item()
        
        dice = (2.0 * intersection) / (union + 1e-8)
        dice_list.append(dice)

    return sum(dice_list) / len(dice_list)   # Retourne la moyenne des coefficients Dice

# Fonction pour remapper un masque en niveaux de gris aux classes du modèle
def remap_mask_to_classes(mask_array):
    # Mapping explicite des niveaux de gris aux indices de classe
    gray_to_class = {
        0: 0,
        15: 1,
        38: 2,
        53: 3,
        75: 4,
        90: 5,
        113: 6,
        128: 7
    }

    remapped_mask = np.zeros_like(mask_array, dtype=np.uint8)

    unknown_values = []

    for gray_value in np.unique(mask_array):
        if gray_value in gray_to_class:
            remapped_mask[mask_array == gray_value] = gray_to_class[gray_value]
        else:
            unknown_values.append(gray_value)
    
    if unknown_values:
        print(f"[WARN] Valeurs inconnues détectées dans le masque : {unknown_values}")

    return remapped_mask

@app.post("/predict/")
async def predict_segmentation(file: UploadFile = File(...), mask_file: UploadFile = None):
    """
    Endpoint pour prédire le masque de segmentation d'une image et calculer les métriques IoU et Dice si un masque est fourni.
    """
    if model is None:
        return {"error": "Modèle non chargé. Veuillez vérifier le chemin du modèle."}

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    original_size = image.size

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    predicted_mask_raw = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Appliquer le remappage au masque prédit
    remapped_predicted_mask = np.zeros_like(predicted_mask_raw, dtype=np.uint8)
    for predicted_class_index, real_class_index in class_mapping.items():
        remapped_predicted_mask[predicted_mask_raw == predicted_class_index] = real_class_index

    # Redimensionner le masque remappé
    remapped_predicted_mask_resized = cv2.resize(
        remapped_predicted_mask,
        (original_size[0], original_size[1]),
        interpolation=cv2.INTER_NEAREST
    )

    iou, dice = None, None

    if mask_file is not None:
        mask_contents = await mask_file.read()
        true_mask_img = Image.open(io.BytesIO(mask_contents))

        # Vérifier le mode du masque et le convertir si nécessaire
        if true_mask_img.mode not in ["L", "P", "1"]:
            true_mask_img = true_mask_img.convert("L")

        # Redimensionner le masque à la taille d'inférence
        true_mask_img_resized = true_mask_img.resize((512, 256), Image.NEAREST)

        true_mask = np.array(true_mask_img_resized)

        # Remap du masque réel aux classes du modèle
        true_mask_remapped = remap_mask_to_classes(true_mask)

        # Pour le calcul du Dice, convertir les deux masques en tenseur PyTorch
        true_mask_tensor = torch.tensor(true_mask_remapped).unsqueeze(0).to(device)

        # Calculer IoU
        iou = calculate_iou(predicted_mask_raw, true_mask_remapped, num_classes=8)

        # Calculer Dice
        dice = calculate_dice_coefficient(output, true_mask_tensor, num_classes=8)

    pixel_counts = {}
    total_pixels = remapped_predicted_mask_resized.size
    for class_idx, class_name in enumerate(class_names):
        count = np.sum(remapped_predicted_mask_resized == class_idx)
        percentage = (count / total_pixels) * 100
        pixel_counts[class_name] = {"count": int(count), "percentage": round(percentage, 2)}

    return {
        "filename": file.filename,
        "image_width": original_size[0],
        "image_height": original_size[1],
        "total_pixels": total_pixels,
        "class_statistics": pixel_counts,
        "iou": iou,
        "dice": dice
    }


@app.post("/predict_with_mask/")
async def predict_with_mask(file: UploadFile = File(...)):
    """
    Endpoint pour prédire et retourner directement l'image du masque avec les couleurs repositionnées.
    """
    if model is None:
        return {"error": "Modèle non chargé. Veuillez vérifier le chemin du modèle."}

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    original_size = image.size

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    predicted_mask_raw = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Appliquer le remappage au masque prédit
    remapped_mask = np.zeros_like(predicted_mask_raw, dtype=np.uint8)
    for predicted_class_index, real_class_index in class_mapping.items():
        remapped_mask[predicted_mask_raw == predicted_class_index] = real_class_index

    # Redimensionner le masque remappé
    remapped_mask_resized = cv2.resize(
        remapped_mask,
        (original_size[0], original_size[1]),
        interpolation=cv2.INTER_NEAREST
    )

    # Créer une image colorée du masque remappé
    colored_mask = np.zeros((remapped_mask_resized.shape[0], remapped_mask_resized.shape[1], 3), dtype=np.uint8)
    for class_idx in range(len(color_palette)):
        colored_mask[remapped_mask_resized == class_idx] = color_palette[class_idx]

    _, encoded_mask = cv2.imencode('.png', colored_mask)
    mask_bytes = encoded_mask.tobytes()

    return Response(content=mask_bytes, media_type="image/png")

@app.post("/overlay_mask/")
async def overlay_mask(file: UploadFile = File(...), alpha: float = 0.5):
    """
    Endpoint pour superposer le masque de segmentation (avec les couleurs repositionnées) sur l'image originale.
    """
    if model is None:
        return {"error": "Modèle non chargé. Veuillez vérifier le chemin du modèle."}

    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    original_size = image_pil.size
    image = np.array(image_pil)  # RGB format

    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    predicted_mask_raw = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Remap des classes comme dans /predict_with_mask/
    remapped_mask = np.zeros_like(predicted_mask_raw, dtype=np.uint8)
    for predicted_class_index, real_class_index in class_mapping.items():
        remapped_mask[predicted_mask_raw == predicted_class_index] = real_class_index

    # Redimensionner le masque remappé
    remapped_mask_resized = cv2.resize(
        remapped_mask,
        (original_size[0], original_size[1]),
        interpolation=cv2.INTER_NEAREST
    )

    # Créer une image colorée du masque remappé (fidèle au 1er endpoint)
    colored_mask = np.zeros((remapped_mask_resized.shape[0], remapped_mask_resized.shape[1], 3), dtype=np.uint8)
    for class_idx in range(len(color_palette)):
        colored_mask[remapped_mask_resized == class_idx] = color_palette[class_idx]

    # Superposer le masque colorisé sur l'image originale
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

    # Pas de conversion RGB → BGR 
    _, encoded_overlay = cv2.imencode('.png', overlay)
    overlay_bytes = encoded_overlay.tobytes()

    return Response(content=overlay_bytes, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)