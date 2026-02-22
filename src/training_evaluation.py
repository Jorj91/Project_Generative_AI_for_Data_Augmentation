import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.metrics import classification_report, accuracy_score


# synthetic dataset

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, class_to_idx, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = class_to_idx

        for idx in metadata:
            for item in metadata[idx]:
                path = item["image_path"]
                class_name = item["class_name"]

                if class_name in class_to_idx:
                    label = class_to_idx[class_name]
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    

# Model

def create_model(num_classes, device):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


# Training

def train_model(model, train_loader, device, epochs=5):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()

    for epoch in range(epochs):
        running_loss = 0

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    return model


# Evaluation

def evaluate_model(model, loader, device):

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, output_dict=True)

    return accuracy, report_dict

# Main Training Pipeline

def run_training(PROJECT_ROOT, epochs=5, batch_size=64):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    # Load datasets

    dataset_train = torchvision.datasets.OxfordIIITPet(
        root=os.path.join(PROJECT_ROOT, "data", "raw"),
        split="trainval",
        transform=transform_train,
        download=True
    )

    dataset_test = torchvision.datasets.OxfordIIITPet(
        root=os.path.join(PROJECT_ROOT, "data", "raw"),
        split="test",
        transform=transform_test,
        download=True
    )

    train_small_idx = np.load(
        os.path.join(PROJECT_ROOT, "data", "splits", "train_small_indices.npy")
    )

    dataset_train_small = Subset(dataset_train, train_small_idx)

    
    # Synthetic data

    metadata_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "synthetic",
        "generation_metadata_final.json"
    )

    with open(metadata_path, "r") as f:
        generation_metadata = json.load(f)

    synthetic_dataset = SyntheticDataset(
        generation_metadata,
        dataset_train.class_to_idx,
        transform=transform_train
    )

    # Baseline vs Augmented

    train_baseline = dataset_train_small

    train_augmented = ConcatDataset([
        dataset_train_small,
        synthetic_dataset
    ])  


    # Loaders
    train_loader_baseline = DataLoader(
        train_baseline,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    train_loader_augmented = DataLoader(
        train_augmented,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Train baseline
    model_baseline = create_model(37, device)
    model_baseline = train_model(model_baseline, train_loader_baseline, device, epochs)

    acc_baseline, report_baseline = evaluate_model(model_baseline, test_loader, device)

    print("Baseline Accuracy:", acc_baseline)

    # Train augmented
    model_augmented = create_model(37, device)
    model_augmented = train_model(model_augmented, train_loader_augmented, device, epochs)
    
    acc_augmented, report_augmented = evaluate_model(model_augmented, test_loader, device)

    print("Augmented Accuracy:", acc_augmented)

    # save results
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # save metrics + experiment info
    results = {
        "baseline": {
            "accuracy": acc_baseline,
            "report": report_baseline
        },
        "augmented": {
            "accuracy": acc_augmented,
            "report": report_augmented
        },
        "experiment_info": {
            "train_size_real": len(train_baseline),
            "train_size_synthetic": len(synthetic_dataset),
            "test_size": len(dataset_test),
            "epochs": epochs,
            "batch_size": batch_size,
            "model": "ResNet18"
        }
    }

    with open(os.path.join(MODEL_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # save models weights
    torch.save(
        model_baseline.state_dict(),
        os.path.join(MODEL_DIR, "resnet18_baseline.pth")
    )

    torch.save(
        model_augmented.state_dict(),
        os.path.join(MODEL_DIR, "resnet18_augmented.pth")
    )

    return results


