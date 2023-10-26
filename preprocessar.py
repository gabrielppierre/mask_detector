import cv2
import os
import numpy as np

def pre_processar():
    com_mascara_path = os.path.join('dataset', 'com_mascara')
    sem_mascara_path = os.path.join('dataset', 'sem_mascara')

    #listas pra os dados e os rotulos
    data = []
    labels = []

    #preprocessamento [com mascara]
    for filename in os.listdir(com_mascara_path):
        img_path = os.path.join(com_mascara_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  #ler em escala de cinza pra economizar dados
        img = cv2.resize(img, (128, 128))  #redimensionar
        img = img / 255.0  #normalizar pra valores entre 0 e 1
        data.append(img)
        labels.append(1)  #rotulo 1 pra 'com mascara'

    #preprocessamento [sem mascara]
    for filename in os.listdir(sem_mascara_path):
        img_path = os.path.join(sem_mascara_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        data.append(img)
        labels.append(0)  #rotulo 0 pra 'sem mascara'

    #transformar em numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    return data, labels
