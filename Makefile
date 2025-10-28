# MAKEFILE

IMAGE_NAME=clip_practica
CONTAINER_NAME=clip_container

# Construir imagen
build:
	docker build -t $(IMAGE_NAME) .

# Ejecutar contenedor interactivo (bash)
run:
	docker run -it --rm -v $(PWD):/app $(IMAGE_NAME) bash

# Ejecutar el script de Pre-CLIP dentro del contenedor
preclip:
	docker run -it --rm -v $(PWD):/app $(IMAGE_NAME) python3 preclip_similarity.py

# Ejecutar el script de CLIP dentro del contenedor
clip:
	docker run -it --rm -v $(PWD):/app $(IMAGE_NAME) python3 clip_similarity.py
