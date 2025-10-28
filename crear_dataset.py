import csv

# === Definir los datos del dataset ===
data = [
    ["image", "description", "category"],

    # --- Category: City Architecture ---
    ["imagen1.png", "several blocks of buildings very close together", "city_architecture"],
    ["imagen2.png", "View of the corner of a glass building.", "city_architecture"],
    ["imagen3.png", "An alley from which several blocks of buildings can be seen", "city_architecture"],
    ["imagen4.png", "two very different building facades", "city_architecture"],
    ["imagen5.png", "circular, glass-covered building with iron figures", "city_architecture"],

    # --- Category: Industrial Areas ---
    ["imagen6.png", "view from the sea of a factory at night", "industrial_areas"],
    ["imagen7.png", "aluminum factory seen at night", "industrial_areas"],
    ["imagen8.png", "four-story factory with yellow stairs", "industrial_areas"],
    ["imagen9.png", "stairs to the office of a white factory", "industrial_areas"],
    ["imagen10.png", "three pipes of a factory seen at night", "industrial_areas"],

    # --- Category: Street Life ---
    ["imagen11.png", "people crossing a crosswalk in new york", "street_life"],
    ["imagen12.png", "people waiting to cross dressed in winter clothes", "street_life"],
    ["imagen13.png", "Two parents and their son lifting him across a pedestrian crossing", "street_life"],
    ["imagen14.png", "The city of Japan seen at night with its inhabitants carrying umbrellas", "street_life"],
    ["imagen15.png", "a small street but full of people and bars", "street_life"],

    # --- Category: Urban Mobility ---
    ["imagen16.png", "the tram running along a main street", "urban_mobility"],
    ["imagen17.png", "the crossing of many roads with cars circulating", "urban_mobility"],
    ["imagen18.png", "people of all ages riding bicycles", "urban_mobility"],
    ["imagen19.png", "a man with a beard and long hair riding his bicycle", "urban_mobility"],
    ["imagen20.png", "A man with an electric scooter and the city tram in front of him", "urban_mobility"],
]

# === Guardar como CSV ===
with open("dataset_UrbanScenes.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(data)

print("CSV creado correctamente: dataset_UrbanScenes.csv")
