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
DATASET_DIR = "dataset"
CAPTIONS_FILE = os.path.join(DATASET_DIR, "captions.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CARGA DE DATOS
df = pd.read_csv(CAPTIONS_FILE)
print(f"Dataset loaded: {len(df)} samples")


# MODELO CLIP
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(model_name)
clip_model.eval()


# EMBEDDINGS
image_embeddings = []
text_embeddings = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    # Imagen
    img_path = os.path.join(DATASET_DIR, row["image_path"])
    image = Image.open(img_path).convert("RGB")
    inputs = clip_processor( # Procesamiento con CLIP 
        text=[row["caption"]],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    with torch.no_grad(): # Obtener embeddings 
        image_embeds = clip_model.get_image_features(inputs["pixel_values"]) 
        text_embeds = clip_model.get_text_features(inputs["input_ids"])

    # Normalización a través de la norma L2 
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    image_embeddings.append(image_embeds.cpu().numpy().flatten())
    text_embeddings.append(text_embeds.cpu().numpy().flatten())

image_embeddings = np.vstack(image_embeddings)
text_embeddings = np.vstack(text_embeddings)

print(f"Embeddings generados con CLIP: {image_embeddings.shape} (imágenes), {text_embeddings.shape} (textos)")



# CÁLCULO DE SIMILITUD

