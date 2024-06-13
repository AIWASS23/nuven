# Primeiro Passo: Construir o CSV catalogando as imagens de acordo com as classes solicitadas
#/ ------------------------------------------------------------------------------------------------------- \#

import os
import pandas as pd

# Caminho para o arquivo
pastaDeDados = 'cells'

# Listar as classes Normais e Anormais
normal_classes = ['Negativa']
anormal_classes = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'carcinoma']

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
            if class_name in normal_classes:
                filenames.append(file)
                classes.append('Normal')
            elif class_name in anormal_classes:
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
    df.to_csv(output_csv, index=False)

    print(f'Tabela salva como {output_csv}')

# Segundo passo: 