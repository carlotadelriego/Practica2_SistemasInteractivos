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

