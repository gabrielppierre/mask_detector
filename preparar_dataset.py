from zipfile import ZipFile
import os
import shutil  # Para remover a pasta e todo o seu conteúdo

def preparar_dataset():
    parent_folder_path = 'dataset'
    
    #nomes das pastas originais e renomeadas
    folders_to_rename = {
        "com_ma╠üscara": "com_mascara",
        "sem_ma╠üscara": "sem_mascara",
    }
    
    #verifica se as pastas ja existem
    new_folders_exist = all(os.path.exists(os.path.join(parent_folder_path, new)) for new in folders_to_rename.values())
    
    #pula se ja existirem
    if new_folders_exist:
        print("Pastas já existem, pulando etapa de extração.")
    else:
        #se nao, extrai o dataset
        if not os.path.exists(parent_folder_path):
            os.makedirs(parent_folder_path)

        with ZipFile('archive.zip', 'r') as dataset:
            dataset.extractall(path=parent_folder_path)

        for original, new in folders_to_rename.items():
            original_path = os.path.join(parent_folder_path, original)
            new_path = os.path.join(parent_folder_path, new)
            
            #remove a pasta se ja existir
            if os.path.exists(new_path):
                shutil.rmtree(new_path)
                
            if os.path.exists(original_path):
                os.rename(original_path, new_path)
            else:
                print(f"A pasta {original} não foi encontrada.")
