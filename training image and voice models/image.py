
# # Library
import os
import random
import shutil
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import random
from PIL import Image, ImageEnhance, ImageFilter
import torch.optim as optim
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Paths to the dataset
DATA_PATH = "data"
OUTPUT_PATH = "processed_data"

# Create directories for processed data
os.makedirs(OUTPUT_PATH, exist_ok=True)
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_PATH, split), exist_ok=True)

# Set random seed for reproducibility
random.seed(42)

# Process each person's folder
for person_id in range(1, 41):  # There are 40 people
    person_folder = os.path.join(DATA_PATH, f"s{person_id}")
    if not os.path.exists(person_folder):
        print(f"Folder {person_folder} not found.")
        continue

    # List all .pgm files for the person
    image_files = [f for f in os.listdir(person_folder) if f.endswith(".pgm")]
    
    # Ensure each person has exactly 10 images
    if len(image_files) != 10:
        print(f"Person {person_id} has {len(image_files)} images instead of 10.")
        continue

    # Shuffle and split the data: 80% train, 10% validation, 10% test
    random.shuffle(image_files)
    train_files = image_files[:8]
    val_file = image_files[8:9]  # Single image for validation
    test_file = image_files[9:]  # Single image for testing

    # Function to copy images to their respective splits
    def copy_files(files, split_name):
        split_path = os.path.join(OUTPUT_PATH, split_name, f"s{person_id}")
        os.makedirs(split_path, exist_ok=True)
        for file in files:
            src_path = os.path.join(person_folder, file)
            dst_path = os.path.join(split_path, file)
            shutil.copy(src_path, dst_path)

    # Copy images
    copy_files(train_files, "train")
    copy_files(val_file, "val")
    copy_files(test_file, "test")

print("Dataset preprocessed and split into train, validation, and test sets.")


# Function to load and plot images from a given folder
def plot_images(person_id, processed_data_path):
    splits = ["train", "val", "test"]
    images = {split: [] for split in splits}
    
    for split in splits:
        folder_path = os.path.join(processed_data_path, split, f"s{person_id}")
        if os.path.exists(folder_path):
            image_files = sorted(os.listdir(folder_path))  # Sort to maintain order
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                images[split].append(image_path)

    # Plot the images
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))  # Adjust grid size if necessary
    fig.suptitle(f"Images for Person {person_id} in Train, Val, and Test Sets", fontsize=16)

    for i, split in enumerate(splits):
        for j, img_path in enumerate(images[split]):
            if j >= 8:  # Limit to max 8 images per row
                break
            ax = axes[i, j]
            img = Image.open(img_path)
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(split.capitalize(), fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate title
    plt.show()

# Example usage
person_id = 1  # Change to the desired person's ID (1 to 40)
processed_data_path = "processed_data"
plot_images(person_id, processed_data_path)



# Define a function for test data preprocessing
transform_val = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.Resize((224, 224)),  # Resize images to 224x224 for ResNet
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    
    #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Blur the image
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, etc.
    
    transforms.Resize((224, 224)),  # Resize images to 224x224 for ResNet
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Define original transform and augmentation transform
original_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

augmentation_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to create augmented versions
def augment_dataset(dataset, num_augmented_per_image=3):
    augmented_images = []
    for img_path, label in dataset.samples:  # Access raw image paths
        img = Image.open(img_path).convert("RGB")  # Open image and ensure 3 channels
        for _ in range(num_augmented_per_image):
            augmented_image = augmentation_transform(img)
            augmented_images.append((augmented_image, label))
    return augmented_images




# Load datasets
original_dataset =  datasets.ImageFolder(root= "processed_data/train", transform=original_transform)
num_augmented_per_image = 2  # Number of augmentations per original image
augmented_images = augment_dataset(original_dataset, num_augmented_per_image)

train_dataset = ConcatDataset([original_dataset, augmented_images])

val_dataset = datasets.ImageFolder(root="processed_data/val", transform=transform_val)  
test_dataset = datasets.ImageFolder(root="processed_data/test", transform=transform_test) # for blurring images or enhancing images modify "transform_test"

# Create DataLoaders
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
valid_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Display dataset sizes
print(f"Train dataset size: {len(train_dataset)} images.")
print(f"Validation dataset size: {len(val_dataset)} images.")
print(f"Test dataset size: {len(test_dataset)} images.")



# Function to denormalize the images (convert back to original pixel range)
def denormalize_image(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.cpu().numpy().transpose((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    tensor = std * tensor + mean  # Denormalize
    tensor = np.clip(tensor, 0, 1)  # Clip values to [0, 1] range
    return tensor

# Function to plot 16 random images
def plot_random_images(loader, title):
    # Check if the loader has enough data
    try:
        images, labels = next(iter(loader))
        if len(images) < 16:
            print(f"Not enough images to plot 16 from {title}!")
            return
    except Exception as e:
        print(f"Error loading images from {title}: {e}")
        return

    images = images[:16]  # Select first 16 images
    labels = labels[:16]  # Select corresponding labels
    
    # Create a grid for displaying images
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = denormalize_image(images[i])  # Denormalize image for display
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Label: {labels[i].item()}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
    plt.show()

# Plot 16 random images from train, val, and test sets
print("Plotting random images from Train Set:")
plot_random_images(train_loader, "Train Set")

print("Plotting random images from Validation Set:")
plot_random_images(valid_loader, "Validation Set")

print("Plotting random images from Test Set:")
plot_random_images(test_loader, "Test Set")


# # train the model

# Load the pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Modify the final fully connected layer to match the number of labels (for CelebA, you have N unique labels)
num_labels = 40  # This is the number of unique labels after filtering
model.fc = nn.Linear(model.fc.in_features, num_labels)

# Freeze all layers except the final fully connected layer
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last fully connected layer to allow it to be trained
for param in model.fc.parameters():
    param.requires_grad = True


# Move the model to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Multi-class classification loss function
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)  # Only optimize the last layer


# Training and validation loop
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10):
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Train loop
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        # Calculate training accuracy
        train_accuracy = correct_train / total_train

        # Validation loop
        model.eval()
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                _, predicted = torch.max(outputs, 1)
                correct_valid += (predicted == labels).sum().item()
                total_valid += labels.size(0)

        # Calculate validation accuracy
        valid_accuracy = correct_valid / total_valid

        # Print loss and accuracy for each epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {valid_accuracy:.4f}")

        # Save the model if validation accuracy is better
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(model.state_dict(), "best_resnet18.pth")





# Call the training function with the training and validation dataloaders
train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=50)



# # EVALUATION
# confusion matrix
def evaluate_model(model, test_loader):
    # Load the best model weights if saved
    model.load_state_dict(torch.load("best_resnet18.pth"))
    model.eval()

    all_preds = []
    all_labels = []

    # No gradient computation during evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Get the predictions
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_labels), yticklabels=range(num_labels))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# Call the function with your test dataloader
evaluate_model(model, test_loader)




# Placeholder for data to save
data_to_save = []

# Loop through your data loader
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)

    outputs = torch.softmax(outputs, dim=1) # apply sofmax
    
    for i in range(len(outputs)):
        # Extract relevant information
        true_label = labels[i].item()
        predicted_label = torch.argmax(outputs[i]).item()
        formatted_tensor = [f"{val:.4f}" for val in outputs[i].tolist()]
        
        # Add to data list as a single row
        row = [true_label, predicted_label] + formatted_tensor
        data_to_save.append(row)
    

# Define column names
num_elements = len(formatted_tensor)  # Number of elements in the tensor
columns = ["True Label", "Predicted Label"] + [f"{i}" for i in range(num_elements)]

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(data_to_save, columns=columns)
df.to_csv("results.csv", index=False)

print("Data saved to 'results.csv'.")


