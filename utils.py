import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import copy

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomGrayscale(),
        transforms.RandomAffine(translate=(0.05, 0.05), degrees=0),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    return train_transform, test_transform

def build_model(num_classes=2):
    model = models.densenet161(pretrained=True)
    
    # Freeze the weights of the pre-trained model
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    initial_num_neurons = model.classifier.in_features
    
    # New classifier
    classifier = nn.Linear(in_features=initial_num_neurons, out_features=num_classes)
    model.classifier = classifier
    
    return model
