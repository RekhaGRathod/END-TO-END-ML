import os
import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import torch.nn.functional as F
from utils import get_device, get_data_transforms, build_model

def get_probs_and_preds(model, loader, device):
    model.eval()
    labels_savings = []
    probs_savings = []
    preds_savings = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(False):
            output = model(images)
            output = F.softmax(output, dim=1)
            probabilities, predictions = torch.max(output, dim=1)

        labels_savings += labels.tolist()
        probs_savings += probabilities.tolist()
        preds_savings += predictions.tolist()

    return labels_savings, probs_savings, preds_savings

def main():
    device = get_device()
    print(f"Using device: {device}")

    _, test_transform = get_data_transforms()
    
    data_dir = 'data/chest_xray/chest_xray'
    test_dir = os.path.join(data_dir, 'test')
    
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found.")
        return

    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
    
    model = build_model(num_classes=2)
    model_path = 'models/best_model.pt'
    
    if not os.path.exists(model_path):
        print("Best model not found. Please run train.py first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    print("Evaluating model...")
    labels, probs, preds = get_probs_and_preds(model, test_loader, device)

    # Calculate metrics
    report = classification_report(labels, preds, target_names=['NORMAL', 'PNEUMONIA'])
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    ticklabels = ['NORMAL', 'PNEUMONIA']
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='.3g', xticklabels=ticklabels, yticklabels=ticklabels, cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.subplot(1, 2, 2)
    sns.heatmap(cmn, annot=True, fmt='.3f', xticklabels=ticklabels, yticklabels=ticklabels, cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    os.makedirs('images', exist_ok=True)
    plt.savefig('images/confusion_matrix.png')
    print("Confusion matrix saved to images/confusion_matrix.png")

if __name__ == '__main__':
    main()
