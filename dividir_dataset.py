import os
import shutil
from sklearn.model_selection import train_test_split

def dividir_dataset():
    parent_folder_path = 'dataset'
    sub_folders = ['com_mascara', 'sem_mascara']
    
    train_dir = os.path.join(parent_folder_path, 'train')
    test_dir = os.path.join(parent_folder_path, 'test')
    val_dir = os.path.join(parent_folder_path, 'val')
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    for folder in sub_folders:
        full_folder_path = os.path.join(parent_folder_path, folder)
        files = [f for f in os.listdir(full_folder_path) if os.path.isfile(os.path.join(full_folder_path, f))]
        train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)
        
        train_folder = os.path.join(train_dir, folder)
        test_folder = os.path.join(test_dir, folder)
        val_folder = os.path.join(val_dir, folder)
        
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        if not os.path.exists(val_folder):
            os.makedirs(val_folder)

        for file in train_files:
            shutil.copy(os.path.join(full_folder_path, file), os.path.join(train_folder, file))
        for file in test_files:
            shutil.copy(os.path.join(full_folder_path, file), os.path.join(test_folder, file))
        for file in val_files:
            shutil.copy(os.path.join(full_folder_path, file), os.path.join(val_folder, file))
