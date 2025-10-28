import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel

# CONFIGURACIÓN
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTIONS_FILE = os.path.join(BASE_DIR, "dataset_UrbanScenes.csv")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CARGA DEL DATASET
df = pd.read_csv(CAPTIONS_FILE)
print(f"Dataset loaded: {len(df)} samples")


# MODELO CLIP
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name).to(DEVICE) # .to(DEVICE) para usar GPU si está disponible
clip_processor = CLIPProcessor.from_pretrained(model_name)
clip_model.eval()


# EXTRACCIÓN DE EMBEDDINGS
image_embeddings = []
text_embeddings = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Procesando muestras"):
    # Ruta completa de la imagen (usa category + image)
    img_path = os.path.join(DATASET_DIR, row["category"], row["image"])
    if not os.path.exists(img_path):
        print(f"Imagen no encontrada: {img_path}")
        continue

    # Procesar con CLIP
    image = Image.open(img_path).convert("RGB")
    inputs = clip_processor(
        text=[row["description"]],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        image_features = clip_model.get_image_features(inputs["pixel_values"])
        text_features = clip_model.get_text_features(inputs["input_ids"])

    # Normalizar embeddings
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    image_embeddings.append(image_features.cpu().numpy().flatten())
    text_embeddings.append(text_features.cpu().numpy().flatten())

# Convertir a arrays
image_embeddings = np.vstack(image_embeddings)
text_embeddings = np.vstack(text_embeddings)
print(f"Embeddings generados con CLIP: {image_embeddings.shape} (imágenes), {text_embeddings.shape} (textos)")


# CÁLCULO DE SIMILITUD
similarity_matrix = cosine_similarity(image_embeddings, text_embeddings)



# RESULTADOS
diagonal_sim = np.diag(similarity_matrix)
mean_correct_sim = np.mean(diagonal_sim)
mean_all_sim = np.mean(similarity_matrix)

print("Resultados de similitud (CLIP):")
print(f"Similitud media entre pares correctos: {mean_correct_sim:.4f}")
print(f"Similitud media global: {mean_all_sim:.4f}")


# VISUALIZACIÓN
plt.figure(figsize=(8, 6))
plt.imshow(similarity_matrix, cmap="plasma")
plt.title("CLIP Similarity Matrix (openai/clip-vit-base-patch32)")
plt.xlabel("Captions")
plt.ylabel("Images")
plt.colorbar(label="Cosine Similarity")
plt.tight_layout()
plt.show()

np.save("clip_similarity_matrix.npy", similarity_matrix)
print("Matriz de similitud guardada en clip_similarity_matrix.npy")
