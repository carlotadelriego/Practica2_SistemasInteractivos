import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA


# CONFIGURACIÓN
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directorio actual
CAPTIONS_FILE = os.path.join(BASE_DIR, "dataset_UrbanScenes.csv")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")  # Directorio con las imágenes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU si está disponible si no CPU

# CARGA DEL DATASET
df = pd.read_csv(CAPTIONS_FILE)
print(f"Dataset loaded: {len(df)} samples") 


# MODELO DE IMAGEN (ResNet50)
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # Usar pesos preentrenados
resnet.fc = torch.nn.Identity()
resnet = resnet.to(DEVICE)
resnet.eval()

# Transformaciones de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# MODELO DE TEXTO
text_model = SentenceTransformer("distiluse-base-multilingual-cased")



# EXTRACCIÓN DE EMBEDDINGS
image_embeddings = []
text_embeddings = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Procesando muestras"):
    # Ruta completa de la imagen (usa category + image)
    img_path = os.path.join(DATASET_DIR, row["category"], row["image"]) # Ajusta según la estructura del dataset
    if not os.path.exists(img_path):
        print(f"Imagen no encontrada: {img_path}")
        continue

    # Imagen a embedding
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = resnet(image_tensor).cpu().numpy().flatten()
    image_embeddings.append(emb)

    # Texto a embedding
    text_emb = text_model.encode(row["description"])
    text_embeddings.append(text_emb)


# Convertir a arrays 
image_embeddings = np.vstack(image_embeddings)
text_embeddings = np.vstack(text_embeddings)
print(f"Embeddings generados: {image_embeddings.shape} (imágenes), {text_embeddings.shape} (textos)")



# CÁLCULO DE SIMILITUD
# Reducir las dimensiones de los embeddings de imagen a 20 con PCA para facilitar el cálculo
pca = PCA(n_components=min(20, image_embeddings.shape[0]))
image_embeddings_reduced = pca.fit_transform(image_embeddings)
print(f"Imágenes reducidas a: {image_embeddings_reduced.shape}")

similarity_matrix = cosine_similarity(image_embeddings_reduced, text_embeddings[:, :image_embeddings_reduced.shape[1]]) # Asegurarse de que las dimensiones coincidan


# Similitud diagonal y medias 
diagonal_sim = np.diag(similarity_matrix) # Similitud entre pares correctos
mean_correct_sim = np.mean(diagonal_sim) # Similitud media entre pares correctos
mean_all_sim = np.mean(similarity_matrix) # Similitud media global

print("Resultados de similitud:")
print(f"Similitud media entre pares correctos: {mean_correct_sim:.4f}")
print(f"Similitud media global: {mean_all_sim:.4f}")


# VISUALIZACIÓN
plt.figure(figsize=(8, 6))
plt.imshow(similarity_matrix, cmap="viridis")
plt.title("Pre-CLIP Similarity Matrix (ResNet + SentenceTransformer)")
plt.xlabel("Captions")
plt.ylabel("Images")
plt.colorbar(label="Cosine Similarity")
plt.tight_layout()
plt.show()


# GUARDADO DE RESULTADOS
np.save("preclip_similarity_matrix.npy", similarity_matrix)
print("Matriz de similitud guardada en preclip_similarity_matrix.npy")
