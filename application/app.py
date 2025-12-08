import streamlit as st
import requests
import io
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Traiter les images d'un système embarqué",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre de l'application
st.title("Visualisation de la Segmentation d'une image")

# URL de l'API
import os

# URL de l'API
# Récupère l'URL depuis les variables d'environnement (pour le Cloud) ou utilise localhost par défaut
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Option pour choisir la transparence du masque superposé
alpha = st.sidebar.slider("Transparence du masque", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Fonction pour télécharger une image et la retourner sous forme d'octets et d'objet PIL
def process_uploaded_image(uploaded_file):
    if uploaded_file is not None:
        # Lire le contenu du fichier en octets
        image_bytes = uploaded_file.getvalue()
        
        # Créer un objet Image PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        return image_bytes, image
    return None, None

# Widget pour télécharger une image
uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])
uploaded_file_mask = st.file_uploader("Choisir le masque de l'image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Traiter l'image téléchargée
    image_bytes, image = process_uploaded_image(uploaded_file)
    
    # Créer deux colonnes pour la première rangée (image originale et statistiques)
    col1_top, col2_top = st.columns(2)
    
    # Afficher l'image originale (en haut à gauche)
    with col1_top:
        st.subheader("Image originale")
        st.image(image, caption="Image téléchargée", use_container_width=True)
    
    with col2_top:
        st.subheader("Masque original")
        if uploaded_file_mask is not None:
            mask_bytes, mask_image = process_uploaded_image(uploaded_file_mask)
            st.image(mask_image, caption="Masque téléchargé", use_container_width=True)
        else:
            st.info("Veuillez télécharger un masque pour l'afficher ici.")
    
    # Créer un espace entre les deux rangées
    st.markdown("---")
    
    # Créer deux colonnes pour la deuxième rangée (masque et superposition)
    col1_bottom, col2_bottom, col3_bottom = st.columns(3)

    # Statistiques de segmentation (en haut à droite)
    with col1_bottom:
        st.subheader("Statistiques de segmentation")
        if st.button("Obtenir les statistiques"):
            try:
                # Appeler l'API endpoint /predict/
                files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
                if uploaded_file_mask is not None:
                    mask_bytes, _ = process_uploaded_image(uploaded_file_mask)
                    files["mask_file"] = (uploaded_file_mask.name, mask_bytes, uploaded_file_mask.type)

                response = requests.post(f"{API_URL}/predict/", files=files)

                if response.status_code == 200:
                    # Récupérer les données JSON
                    result = response.json()

                    # Afficher les métriques IoU et Dice si disponibles
                    if result['iou'] is not None:
                        st.write(f"IoU: {result['iou']:.4f}")
                    else:
                        st.write("IoU: Non disponible (pas de masque fourni)")

                    if result['dice'] is not None:
                        st.write(f"Dice: {result['dice']:.4f}")
                    else:
                        st.write("Dice: Non disponible (pas de masque fourni)")

                    # Créer un DataFrame pour les statistiques des classes
                    class_stats = result['class_statistics']
                    df_stats = pd.DataFrame({
                        'Classe': list(class_stats.keys()),
                        'Pixels': [class_stats[cls]['count'] for cls in class_stats],
                        'Pourcentage (%)': [class_stats[cls]['percentage'] for cls in class_stats]
                    })

                    # Afficher le tableau
                    st.dataframe(df_stats)

                    # Créer un graphique à barres pour les pourcentages
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(df_stats['Classe'], df_stats['Pourcentage (%)'])

                    # Ajouter les valeurs sur les barres
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                f'{height:.1f}%', ha='center', va='bottom')

                    ax.set_xlabel('Classe')
                    ax.set_ylabel('Pourcentage (%)')
                    ax.set_title('Répartition des classes')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    # Afficher le graphique
                    st.pyplot(fig)
                else:
                    st.error(f"Erreur: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Erreur lors de la connexion à l'API: {str(e)}")
    
    # Bouton pour obtenir le masque de segmentation (en bas à gauche)
    with col2_bottom:
        st.subheader("Masque de segmentation")
        if st.button("Obtenir le masque"):
            try:
                # Appeler l'API endpoint /predict_with_mask/
                files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
                response = requests.post(f"{API_URL}/predict_with_mask/", files=files)
                
                if response.status_code == 200:
                    # Convertir les octets en image
                    mask_image = Image.open(io.BytesIO(response.content))
                    
                    # Afficher l'image du masque
                    st.image(mask_image, caption="Masque de segmentation", use_container_width=True)
                    
                    # Légende des couleurs
                    st.subheader("Légende des classes")
                    colors = [
                        [0, 0, 0],        # Classe 0 - Void
                        [128, 0, 0],      # Classe 1 - Flat
                        [0, 128, 0],      # Classe 2 - Construction
                        [128, 128, 0],      # Classe 3 - Object
                        [0, 0, 128],    # Classe 4 - Nature
                        [128, 0, 128],    # Classe 5 - Sky
                        [0, 128, 128],    # Classe 6 - Human
                        [128, 128, 128]   # Classe 7 - Vehicle
                    ]

                    class_names = [
                        "Void", "Flat", "Construction", "Object", 
                        "Nature", "Sky", "Human", "Vehicle"
                    ]
                    
                    # Noms lisibles des couleurs
                    color_names = [
                        "Noir",        # [0, 0, 0]
                        "Bleu foncé",  # [128, 0, 0]
                        "Vert",        # [0, 128, 0]
                        "Cyan",       # [128, 128, 0]
                        "Rouge",       # [0, 0, 128]
                        "Mauve",      # [128, 0, 128]
                        "Jaune",        # [0, 128, 128]
                        "Gris"         # [128, 128, 128]
                    ]

                    # Créer un DataFrame pour la légende
                    color_df = pd.DataFrame({
                        'Classe': class_names,
                        'Couleur': color_names
                    })
                    
                    # Afficher la légende sous forme de tableau
                    st.dataframe(color_df)
                else:
                    st.error(f"Erreur: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Erreur lors de la connexion à l'API: {str(e)}")
    
    # Bouton pour obtenir l'image avec le masque superposé (en bas à droite)
    with col3_bottom:
        st.subheader("Masque superposé")
        if st.button("Obtenir l'image superposée"):
            try:
                # Appeler l'API endpoint /overlay_mask/
                files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
                response = requests.post(f"{API_URL}/overlay_mask/", params={"alpha": alpha}, files=files)
                
                if response.status_code == 200:
                    # Convertir les octets en image
                    overlay_image = Image.open(io.BytesIO(response.content))
                    
                    # Afficher l'image superposée
                    st.image(overlay_image, caption=f"Image avec masque superposé (alpha={alpha})", use_container_width=True)
                else:
                    st.error(f"Erreur: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Erreur lors de la connexion à l'API: {str(e)}")

# Ajouter des informations sur l'application
st.sidebar.markdown("---")
st.sidebar.header("À propos")
st.sidebar.info("""
Cette application permet de visualiser les résultats d'un modèle de segmentation visuelle VGG16.
Les images sont traitées par une API pour identifier 8 catégories différentes: 
Void, Flat, Construction, Object, Nature, Sky, Human, Vehicle.
""")

st.sidebar.markdown("---")
st.sidebar.header("Instructions")
st.sidebar.info("""
1. Téléchargez une image via le sélecteur de fichiers
2. Ajouter le masque original si vous souhaitez obtenir les métriques de segmentation
3. Utilisez les boutons pour obtenir différentes visualisations :
   - Statistiques de segmentation : pourcentages et nombre de pixels par classe
   - Masque de segmentation : visualisation du masque coloré
   - Masque superposé : superposition du masque sur l'image originale
4. Ajustez la transparence du masque superposé avec le curseur
""")