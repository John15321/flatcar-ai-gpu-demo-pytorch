"""
Module for training and evaluating a neural network on the Fashion-MNIST
dataset using PyTorch and TensorBoard. Provides CLI commands for
training and inference.
"""

import argparse
import os
import random

import torch
import torchvision
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

class_labels = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


class SimpleNN(nn.Module):
    """A simple fully connected neural network for classification."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass of the network."""
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def predict(image_path, model_path):
    """Predict the class of a given image using a trained model."""
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(image.to(device))
        _, predicted = torch.max(output, 1)

    predicted_class = predicted.item()
    print(f"Predicted class: {predicted_class} ({class_labels[predicted_class]})")


def train_and_evaluate(  # pylint: disable=too-many-locals
    batch_size=64,
    learning_rate=0.001,
    epochs=5,
    log_dir="runs/fashion_mnist",
    model_path="fashion_mnist.pth",
):
    """
    Train and evaluate a neural network on the Fashion-MNIST dataset.

    Args:
        batch_size (int): Number of samples per batch.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        log_dir (str): Directory for TensorBoard logs.
        model_path (str): Path to save the trained model.
    """
    print("Initializing training...")
    print(f"Batch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {epochs}")
    print(f"Logging to: {log_dir}")
    print(f"Model will be saved to: {model_path}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=transform, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
        )
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_dir)

    print("Starting training...")
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                writer.add_scalar(
                    "Training Loss", running_loss / 100, epoch * len(train_loader) + i
                )
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        writer.add_scalar("Test Accuracy", accuracy, epoch)
        print(f"Epoch {epoch+1} completed - Test Accuracy: {accuracy:.2f}%")

    print("Saving model...")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print("Training complete!")
    writer.close()


def download_fashion_mnist_samples(output_dir="fashion_mnist_samples", num_samples=10):
    """
    Download and save randomly selected sample images from the Fashion-MNIST dataset,
    ensuring variety.
    """
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=transform, download=True
    )

    # Ensure we get at least one image from each class
    images_per_class = {i: [] for i in range(len(class_labels))}
    for image, label in dataset:
        if len(images_per_class[label]) < num_samples // len(class_labels):
            images_per_class[label].append(image)
        if sum(len(v) for v in images_per_class.values()) >= num_samples:
            break

    # Flatten the dictionary values and shuffle
    selected_images = [
        (image, label) for label, images in images_per_class.items() for image in images
    ]
    random.shuffle(selected_images)

    # Save the selected images
    for i, (image, label) in enumerate(selected_images[:num_samples]):
        file_name = os.path.join(
            output_dir, f"fashion_mnist_{label}_{i}_{class_labels[label]}.png"
        )
        img = Image.fromarray((image.numpy().squeeze() * 255).astype("uint8"))
        img.save(file_name)
        print(f"Saved: {file_name} (Label: {label}) ({class_labels[label]})")


def main_download_fashion_mnist_samples():
    """CLI entry point for downloading Fashion-MNIST dataset samples."""
    parser = argparse.ArgumentParser(
        description="Download Fashion-MNIST dataset sample images."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="fashion_mnist_samples",
        help="Directory to save Fashion-MNIST samples.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of samples to download."
    )

    args = parser.parse_args()
    download_fashion_mnist_samples(
        output_dir=args.output_dir, num_samples=args.num_samples
    )


def main_train():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train a model on Fashion-MNIST.")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs/fashion_mnist",
        help="Directory for TensorBoard logs.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="fashion_mnist.pth",
        help="Path to save the trained model.",
    )

    args = parser.parse_args()
    train_and_evaluate(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        log_dir=args.log_dir,
        model_path=args.model_path,
    )


def main_predict():
    """CLI entry point for making predictions on an image using a trained model."""
    parser = argparse.ArgumentParser(
        description="Predict an image class using a trained model."
    )
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="fashion_mnist.pth",
        help="Path to the trained model.",
    )

    args = parser.parse_args()
    predict(args.image_path, args.model_path)
