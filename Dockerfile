### DOCKERFILE

FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# Copiar archivos del proyecto
COPY . /app

# Instalar dependencias básicas
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install --no-cache-dir torch torchvision torchaudio \
    transformers \
    sentence-transformers \
    scikit-learn \
    matplotlib \
    pandas \
    pillow \
    tqdm

# Comando por defecto (se puede cambiar con el Makefile)
CMD ["bash"]