import numpy as np
from tensorflow.keras.preprocessing import image
import nuven


def preprocess_image(img_path, target_size=(50, 50)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar uma dimensão extra para o batch
    return img_array

def classify_cell(model, img_path):
    # Pré-processar a imagem
    img_array = preprocess_image(img_path)
    
    # Fazer a predição
    predictions = model.predict(img_array)
    
    # Obter a classe prevista
    predicted_class = np.argmax(predictions, axis=1)
    
    # Mapear o índice da classe para o nome da classe
    class_labels = {0: 'Normal', 1: 'Anormal'}
    return class_labels[predicted_class[0]]

# Caminho para a imagem a ser classificada
image_path = 'araujo.tif'

# Classificar a célula
result = classify_cell(nuven.model1, image_path)
print(f"A célula é classificada como: {result}")