import hashlib
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras import Input

# Caminho para o arquivo
pastaDeDados = 'cells'

# Listar as classes Normais e Anormais
normalClasses = ['Negativa']
anormalClasses = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'carcinoma']

# Inicializar listas para armazenar os nomes das imagens e suas classes
filenames = []
classes = []

# Caminhar pelas pastas e arquivos extraídos
for root, dirs, files in os.walk(pastaDeDados):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            # Obter o nome da classe a partir do caminho
            class_name = os.path.basename(root)
            
            # Classificar as imagens como Normal ou Anormal
            if class_name in normalClasses:
                filenames.append(file)
                classes.append('Normal')
            elif class_name in anormalClasses:
                filenames.append(file)
                classes.append('Anormal')

# Verificar se encontramos arquivos
if not filenames:
    print("Nenhum arquivo de imagem foi encontrado.")
else:
    # Criar a tabela com duas colunas: filename e class
    data = {
        'filename': filenames,
        'class': classes
    }

    # Converter para DataFrame
    df = pd.DataFrame(data)

    # Salvar a tabela em um arquivo CSV
    output_csv = 'cell_images_classification.csv'
    df.to_csv(output_csv, index = False)

    print(f'Tabela salva como {output_csv}')

# Verificar se cada imagem tem um rótulo

csv_file = 'cell_images_classification.csv'
df_csv = pd.read_csv(csv_file)

image_files = []

# Caminhar pelas pastas e subpastas para encontrar os arquivos de imagem
for root, dirs, files in os.walk(pastaDeDados):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            image_files.append(file)


# Verificar se cada imagem está presente no CSV
for filename in image_files:
    if filename not in df_csv['filename'].values:
        print(f"A imagem {filename} não tem uma classe associada no CSV.")

# Função para calcular o hash de um arquivo de imagem
def calculate_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Inicializar um dicionário para armazenar os hashes e nomes dos arquivos
file_hashes = {}
duplicate_files = []

# Caminhar pelas pastas e subpastas para encontrar os arquivos de imagem
for root, dirs, files in os.walk(pastaDeDados):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            file_path = os.path.join(root, file)
            file_hash = calculate_file_hash(file_path)
            
            if file_hash in file_hashes:
                duplicate_files.append((file_path, file_hashes[file_hash]))
            else:
                file_hashes[file_hash] = file_path

# Exibir, se existir, os arquivos duplicados
if duplicate_files:
    print("Arquivos duplicados encontrados:")
    for duplicate in duplicate_files:
        print(f"Duplicado: {duplicate[0]} é duplicata de {duplicate[1]}")
else:
    print("Nenhum arquivo duplicado encontrado.")

shape = 50
activation = 'relu6'

def load_data(main_folder, labels_file, target_size=(shape, shape)): 
    # Carregar o arquivo CSV com as labels
    labels_df = pd.read_csv(labels_file)
    
    # Lista para armazenar os caminhos das imagens e as labels correspondentes
    images = []
    labels = []

    # Mapeamento das labels para números
    label_mapping = {label: idx for idx, label in enumerate(labels_df['class'].unique())}
    
    # Iterar sobre as imagens na pasta principal e subpastas
    for root, dirs, files in os.walk(main_folder):
        for filename in files:
            if filename.endswith(".tif"):
                # Caminho completo para a imagem
                image_path = os.path.join(root, filename)
                
                # Carregar a imagem e redimensioná-la
                img = image.load_img(image_path, target_size=target_size)
                img_array = image.img_to_array(img)
                
                # Adicionar a imagem ao conjunto de dados
                images.append(img_array)
                
                # Extrair a label correspondente do arquivo CSV
                label = labels_df[labels_df['filename'] == filename]['class'].values[0]
                label_idx = label_mapping[label]
                labels.append(label_idx)
    
    # Converter para arrays numpy
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels, label_mapping

# Carregar todas as imagens e labels
images, labels, label_mapping = load_data(pastaDeDados, csv_file)

# Dividir os dados em treino e teste
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalizar os dados
train_images = train_images / 255.0
test_images = test_images / 255.0

# Obter o número de classes
num_classes = len(label_mapping)

# Converter as labels para one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)

# Definir os modelos
model1 = Sequential()
model1.add(Input(shape=(shape, shape, 3)))
model1.add(Conv2D(16, kernel_size=(3, 3), activation = activation))
model1.add(Conv2D(32, (3, 3), activation = activation))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, (3, 3), activation = activation))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, (3, 3), activation = activation))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, (3, 3), activation =activation))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))
model1.add(Flatten())
model1.add(Dense(32, activation = activation))
model1.add(Dropout(0.5))
model1.add(Dense(num_classes, activation='softmax'))

# Compilar os modelos
model1.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy']) # https://keras.io/api/optimizers/

# Treinar os modelos
history1 = model1.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

# Avaliar os modelos nos dados de teste
results_model1_test = model1.evaluate(test_images, test_labels)
results_model1_train = model1.evaluate(train_images, train_labels)

# Imprimir os resultados
print("Resultados do modelo:")
print(f"Acurácia teste com {activation}: {results_model1_test[1]}")
print(f"Acurácia treino com {activation}: {results_model1_train[1]}")
print(f"Perda teste com {activation}: {results_model1_test[0]}")
print(f"Perda treino com {activation}: {results_model1_train[0]}")

# Previsões nos dados de teste
predictions = model1.predict(test_images)

# Converter previsões e labels para classe
predictions_classes = np.argmax(predictions, axis=1)
test_labels_classes = np.argmax(test_labels, axis=1)

precision = precision_score(test_labels_classes, predictions_classes, average = 'weighted')
recall = recall_score(test_labels_classes, predictions_classes, average = 'weighted')

print(f"Precisão: {precision}")
print(f"Revocação: {recall}")

# Matriz de Confusão
cm1 = confusion_matrix(test_labels_classes, predictions_classes)

# Exibir Matriz de Confusão
def plot_confusion_matrix(cm, model_name):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anormal'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.savefig(f"Matriz de Confusão - {model_name}")
    plt.show()
    
plot_confusion_matrix(cm1, activation)