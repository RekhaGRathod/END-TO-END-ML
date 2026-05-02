import os
import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import get_device, get_data_transforms, build_model

def training_step(model, loader, loss_function, optimizer, device):
    model.train()
    epoch_loss = 0
    epoch_correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            output = model(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            _, predictions = torch.max(output, dim=1)

        epoch_loss += loss.item() * images.size(0)
        epoch_correct += torch.sum(predictions == labels)

    epoch_loss = epoch_loss / len(loader.dataset)
    accuracy = epoch_correct.double() / len(loader.dataset)
    return epoch_loss, accuracy

def evaluate_model(model, loader, loss_function, device):
    model.eval()
    epoch_loss = 0
    epoch_correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(False):
            output = model(images)
            loss = loss_function(output, labels)
            _, predictions = torch.max(output, dim=1)

        epoch_loss += loss.item() * images.size(0)
        epoch_correct += torch.sum(predictions == labels)

    epoch_loss = epoch_loss / len(loader.dataset)
    accuracy = epoch_correct.double() / len(loader.dataset)
    return epoch_loss, accuracy

def main():
    print("Checking for dataset...")
    
    data_dir = 'data/chest_xray/chest_xray'
    if not os.path.exists(data_dir):
        print(f"Directory '{data_dir}' not found.")
        print("Please download the 'Chest X-Ray Images (Pneumonia)' dataset from Kaggle,")
        print("unzip it, and place the 'chest_xray' folder inside the 'data' folder.")
        return

    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('images', exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_transform, _ = get_data_transforms()
    
    try:
        data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
        train_split, val_split = train_test_split(data, test_size=0.3)
    except FileNotFoundError:
        print("Dataset not found. Ensure the dataset structure is correct.")
        return

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=batch_size, shuffle=True)

    model = build_model(num_classes=2)
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    epochs = 15
    best_val_loss = float('inf')

    train_loss_history, train_acc_history = [], []
    val_loss_history, val_acc_history = [], []

    for epoch in range(epochs):
        train_loss, train_acc = training_step(model, train_loader, loss_function, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, loss_function, device)

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc.item())
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc.item())

        print(f'Epoch: {epoch+1:02}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_model.pt')
            print("Model saved!")

        scheduler.step()

    # Plot training metrics
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train')
    plt.plot(val_loss_history, label='Val')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train')
    plt.plot(val_acc_history, label='Val')
    plt.title('Accuracy')
    plt.legend()

    plt.savefig('images/training_plot.png')
    print("Training finished and plot saved to images/training_plot.png")

if __name__ == '__main__':
    main()
