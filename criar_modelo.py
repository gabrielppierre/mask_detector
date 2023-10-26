import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime

#definindo a arquitetura do modelo
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3) #Uma camada convolucional que recebe uma imagem com 3 canais de cores (imagem RGB)
        #e aplica 16 filtros diferentes usando um kernel (janela de convolução) de tamanho 3x3.
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 2)#pegar a saída da camada anterior e transformar em um vetor de tamanho 2 (um pra cada classe)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 32 * 6 * 6)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

def avaliar_modelo(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

def criar_e_treinar_modelo():
    #criar pasta com data e hora
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(current_time)
    
    #dados e transformações
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    train_data = datasets.ImageFolder(root=r'dataset/train', transform=transform)
    val_data = datasets.ImageFolder(root=r'dataset/val', transform=transform)
    test_data = datasets.ImageFolder(root=r'dataset/test', transform=transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    #modelo, otimizador, função de perda
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    highest_accuracy = 0  #para salvar só o modelo com maior acurácia
    
    #treinamento e validaçao
    for epoch in range(50):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / len(val_loader.dataset)
        print(f"Epoch {epoch}, Val loss: {val_loss}, Accuracy: {accuracy}%")
        
        #salvar o modelo com a maior acuracia
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            torch.save(model.state_dict(), f"{current_time}/model_highest_accuracy.pt")
    
    #avaliar o modelo com a maior acuracia
    model.load_state_dict(torch.load(f"{current_time}/model_highest_accuracy.pt"))
    avaliar_modelo(model, test_loader, criterion)

if __name__ == '__main__':
    criar_e_treinar_modelo()
