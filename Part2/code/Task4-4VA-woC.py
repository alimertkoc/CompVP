import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from LimitedColorPerceptionDataset import LimitedColorPerceptionDataset
from VisualAcuityDataset import VisualAcuityDataset
import os

# Define transformation parameters for stages
def get_transform(stage):
    if stage == 1:
        # Stage 1: High blur, limited color perception
        return transforms.Compose([
            transforms.Resize((64, 64)),
            VisualAcuityDataset(month_age=6),  # Limited color depth   
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif stage == 2:
        # Stage 2: Medium blur, improved color perception
        return transforms.Compose([
            transforms.Resize((64, 64)),
            VisualAcuityDataset(month_age=3),  
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif stage == 3:
        # Stage 3: Minimal blur, full color perception
        return transforms.Compose([
            transforms.Resize((64, 64)),
            VisualAcuityDataset(month_age=0),  
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

# Load Tiny ImageNet dataset
def load_data(stage, batch_size=32):
    transform = get_transform(stage)
    train_dataset = datasets.ImageFolder(root='tiny-imagenet-200/train', transform=transform)
    val_dataset = datasets.ImageFolder(root='tiny-imagenet-200/val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

# Initialize ResNet34 model
def get_model():
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 200)  # Tiny ImageNet has 200 classes
    return model

# Training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

# Main execution
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stages = [1, 2, 3]
    num_epochs_per_stage = 5
    batch_size = 32

    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for stage in stages:
        print(f'Training Stage {stage}')
        train_loader, val_loader = load_data(stage, batch_size)
        dataloaders = {'train': train_loader, 'val': val_loader}

        model = train_model(model, dataloaders, criterion, optimizer, num_epochs_per_stage, device)

    # Create the networks directory if it doesn't exist
    os.makedirs('networks', exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), 'networks/task4-4VA-woC.pth')

if __name__ == '__main__':
    main()
