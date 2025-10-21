import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image

# ===============================
# CONFIGURACIÓN DE DIRECTORIOS
# ===============================

# Carpeta base donde están las imágenes a clasificar
predict_folder = "dataset/predict"
cat_folder = os.path.join(predict_folder, "cat")
nocat_folder = os.path.join(predict_folder, "nocat")

# Crear carpetas si no existen
os.makedirs(cat_folder, exist_ok=True)
os.makedirs(nocat_folder, exist_ok=True)

# ===============================
# FUNCIONES AUXILIARES
# ===============================

def load_and_preprocess_image(path):
    """Carga una imagen, la convierte a 64x64 RGB, aplana y normaliza"""
    img = Image.open(path).convert("RGB").resize((64, 64))
    img_array = np.array(img) / 255.0  # Normaliza
    return img_array.flatten().reshape(1, -1)  # (1, 12288)

# ===============================
# IMPORTAR LA RED ENTRENADA
# ===============================

# Asegúrate de que NeuralNetwork esté importada desde tu código anterior.
# Aquí solo se asume que ya está definida la clase `NeuralNetwork`.

# Carga de pesos entrenados (si los guardaste en un archivo .npz)
# Por ejemplo, supón que al final del entrenamiento hiciste algo como:
# np.savez("model_weights.npz", w1=nn.w1, b1=nn.b1, w2=nn.w2, b2=nn.b2)

model = NeuralNetwork(input_size=64*64*3, hidden_size=128, output_size=1, learning_rate=0.001)

# Cargar pesos guardados (si existen)
if os.path.exists("model_weights.npz"):
    weights = np.load("model_weights.npz")
    model.w1 = weights["w1"]
    model.b1 = weights["b1"]
    model.w2 = weights["w2"]
    model.b2 = weights["b2"]
    print("✅ Pesos del modelo cargados correctamente.")
else:
    print("⚠️ No se encontró 'model_weights.npz'. Asegúrate de entrenar y guardar el modelo antes.")

# ===============================
# CLASIFICAR CADA IMAGEN
# ===============================

# Listar las imágenes dentro de dataset/predict (sin las subcarpetas)
images = [f for f in os.listdir(predict_folder)
          if f.lower().endswith((".png", ".jpg", ".jpeg"))]

for img_name in images:
    img_path = os.path.join(predict_folder, img_name)
    
    # Preprocesar imagen
    img_data = load_and_preprocess_image(img_path)
    
    # Predecir usando la red
    prediction = model.predict(img_data)
    
    # Clasificación binaria
    label = "cat" if prediction > 0.5 else "nocat"
    
    # Mover la imagen a la carpeta correspondiente
    if label == "cat":
        shutil.move(img_path, os.path.join(cat_folder, img_name))
        print(f"🐱 {img_name} → CAT")
    else:
        shutil.move(img_path, os.path.join(nocat_folder, img_name))
        print(f"🚫 {img_name} → NOCAT")

print("\n✅ Clasificación completa.")
print(f"Imágenes de gatos: {len(os.listdir(cat_folder))}")
print(f"Imágenes de no gatos: {len(os.listdir(nocat_folder))}")
