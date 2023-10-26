# interface.py
import tkinter as tk
from tkinter import filedialog
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from criar_modelo import SimpleCNN  # Certifique-se de que esse import está correto

def inferencia_camera(model_state_dict_path):
    #inicializar o modelo e carregar os pesos
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_state_dict_path))
    model.eval() #modo de avaliação

    cap = cv2.VideoCapture(0)

    #transformações para o frame
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    #carregar o classificador de faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        #capturar o frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 8)

        for (x, y, w, h) in faces:
            face_img = Image.fromarray(cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB))
            image_tensor = transform(face_img).unsqueeze(0)

            #inferencia
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = output.max(1)

            #desenhar o retangulo e o texto
            label = "Com mascara" if predicted.item() == 0 else "Sem mascara"
            color = (0, 255, 0) if label == "Com mascara" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('detector de mascara', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def fechar_programa():
    root.quit()
    root.destroy()

def abrir_video():
    filepath = filedialog.askopenfilename()
    if not filepath:
        return

    #nicializar o modelo e carregar os pesos
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_state_dict_path))
    model.eval()

    cap = cv2.VideoCapture(filepath)
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_img = Image.fromarray(cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB))
            image_tensor = transform(face_img).unsqueeze(0)

            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = output.max(1)

            label = "Com mascara" if predicted.item() == 0 else "Sem mascara"
            color = (0, 255, 0) if label == "Com mascara" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('detector de mascara', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

root = tk.Tk()
root.title("Detecção com Câmera ou Vídeo")

model_state_dict_path = r"2023-10-25_16-20-52\model_highest_accuracy_0.9861111111111112.pt"

btn_camera = tk.Button(root, text="Abrir Câmera", command=lambda: inferencia_camera(model_state_dict_path))
btn_camera.pack(side="left", padx=20, pady=20)

btn_video = tk.Button(root, text="Abrir Vídeo", command=abrir_video)
btn_video.pack(side="right", padx=20, pady=20)

btn_fechar = tk.Button(root, text="Fechar Programa", command=fechar_programa)  # Botão para fechar o programa
btn_fechar.pack(side="bottom", padx=20, pady=20)

root.mainloop()



if __name__ == "__main__":
    model_state_dict_path = r"2023-10-25_16-20-52\model_highest_accuracy_0.9861111111111112.pt" 
    inferencia_camera(model_state_dict_path)
