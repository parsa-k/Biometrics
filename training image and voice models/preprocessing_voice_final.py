
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt





# Path to the data directory
data_dir = './data'

# List all folders (people) in the data directory
people_folders = os.listdir(data_dir)

# Initialize an empty list to hold the data
data = []

# Loop through each folder (person)
for label, person_folder in enumerate(people_folders):
    # Path to the current person's folder
    person_folder_path = os.path.join(data_dir, person_folder)
    
    # Ensure it's a directory
    if os.path.isdir(person_folder_path):
        # List all audio files in the current person's folder
        audio_files = os.listdir(person_folder_path)
        
        # Loop through each audio file and create a record with the label (person) and filename
        for audio_file in audio_files:
            # Create the file path
            file_path = os.path.join(person_folder_path, audio_file)
            # Append the filename and label to the data list
            data.append([file_path, label])

# Convert the list to a DataFrame
df = pd.DataFrame(data, columns=['file_path', 'label'])

# Show the first few records
print(df.head())



# Step 1: Read data and prepare labels and file paths
def get_subfolder_names(main_folder_path):
    file_names = []
    subfolder_names = []
    for folder_ID in os.listdir(main_folder_path):
        subfolder_path = os.path.join(main_folder_path, folder_ID)
        for file_ID in os.listdir(subfolder_path):
            if os.path.isdir(subfolder_path):
                subfolder_names.append(folder_ID)
            if os.path.isfile(os.path.join(subfolder_path, file_ID)):
                file_names.append(file_ID)
    return subfolder_names, file_names

# Load all subfolder and file names into a DataFrame
main_folder_path = "C:/Users/ryan_/OneDrive/Documents/USF/Courses/Fall 2024/Mobile Biometrics/Project/Voice Recognition/data" # Edit this path accordingly, ensure that this is the correct directory for the "data" folder
subfolders, file_names = get_subfolder_names(main_folder_path)
data = {'Label': subfolders, 'File Name': file_names}
df = pd.DataFrame(data)


# Generate spectrograms and split the dataset
# Artificially adds noise to the audio files
def add_noise_to_audio(audio, noise_factor):
    """
    Adds Gaussian noise to the audio signal.
    
    Parameters:
    - audio: The raw audio signal (numpy array)
    - noise_factor: Standard deviation of the Gaussian noise to be added
    
    Returns:
    - Noisy audio signal (numpy array)
    """
    noise = np.random.normal(0, noise_factor, audio.shape)
    noisy_audio = audio + noise
    # Clip the noisy audio to prevent values from going out of the valid range [-1, 1]
    noisy_audio = np.clip(noisy_audio, -1, 1)
    return noisy_audio


# Object that defines the dataset to be used for training, validation, and testing
class VoiceDataset(Dataset):
    def __init__(self, df, main_folder_path, transform=None, noise_factor=0): # The noise factor is input here
        self.df = df
        self.main_folder_path = main_folder_path
        self.transform = transform
        self.noise_factor = noise_factor  # Parameter to control the amount of noise

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['Label']
        file = row['File Name']

        filepath = os.path.join(self.main_folder_path, label, file)

        label = int(label)-1
        y, sr = librosa.load(filepath, sr=None)

        # Ensure the audio is at least 1 second long
        target_length = sr * 1  # 1 second of audio
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]

        # Add noise to the audio signal
        noisy_audio = add_noise_to_audio(y, self.noise_factor)

        # Perform Short-Time Fourier Transform (STFT)
        D = librosa.stft(noisy_audio)  # Use noisy audio here
        DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Convert the spectrogram to a tensor and normalize if needed
        spectrogram = torch.tensor(DB, dtype=torch.float32)

        # Normalize the spectrogram (ResNet18 normalization)
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()
        spectrogram = torch.nn.functional.interpolate(spectrogram.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        return spectrogram, label


# finiding labels and files 
people_data = {}
for _, row in df.iterrows():
    label = row['Label']
    file = row['File Name']
    filepath = os.path.join(main_folder_path, label, file)
    
    if label not in people_data:
        people_data[label] = []
    people_data[label].append(filepath)


# Use all the available people (folders)
# Since labels are already the folder names, we do not need a separate mapping
labels = list(people_data.keys())

# Split files for each person into train, val, and test sets
train_files, val_test_files = [], []
train_label, val_test_label = [], []

for person in labels:
    person_files = []
    person_fil = people_data[person]
    for i in range(500):
       if i % 4 == 0:
           person_files.append(person_fil[i])

    train_size = int(0.8 * len(person_files))
    
    # Split data for each person
    train_files.extend(person_files[:train_size])
    train_label.extend([person]*(train_size))  # Use folder name (string) as label
    val_test_files.extend(person_files[train_size:])
    val_test_label.extend([person]*(len(person_files) - train_size))  # Use folder name (string) as label
    
# Further split the remaining 10% into val and test sets
val_files, test_files, val_label, test_label = train_test_split(val_test_files, val_test_label, test_size=0.5)

# Combine the files into DataFrames using the label directly
train_df = pd.DataFrame({'File Name': train_files, 'Label': train_label})
val_df = pd.DataFrame({'File Name': val_files, 'Label': val_label})
test_df = pd.DataFrame({'File Name': test_files, 'Label': test_label})


# Create DataLoaders for training, validation, and test sets
# Define a transform for ResNet18 preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
])

train_dataset = VoiceDataset(train_df, main_folder_path, transform=transform)
val_dataset = VoiceDataset(val_df, main_folder_path, transform=transform)
test_dataset = VoiceDataset(test_df, main_folder_path, transform=transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# The datasets and dataloaders are now ready to be used with a model for training



# Plots the spectograms 
def plot_spectrograms(dataloader, title, n_images=16):
    # Set up the plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))  # 4x4 grid of subplots
    axes = axes.flatten()  # Flatten the axes for easy iteration

    # Iterate through the dataloader and plot 16 images
    for i, (spectrograms, labels) in enumerate(dataloader):
        if i >= 1:  # We only need the first batch
            break

        # Randomly select 16 spectrograms from the batch
        random_indices = np.random.choice(len(spectrograms), size=n_images, replace=False)

        for j, idx in enumerate(random_indices):
            ax = axes[j]
            spec = spectrograms[idx].numpy()

            # Plot the spectrogram
            ax.imshow(spec, aspect='auto', cmap='inferno', origin='lower')
            ax.set_title(f'Label: {labels[idx]}')
            ax.axis('off')  # Hide axis ticks and labels

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


plot_spectrograms(train_loader, "Training Set: 16 Spectrograms")

plot_spectrograms(val_loader, "Validation Set: 16 Spectrograms")

plot_spectrograms(test_loader, "Test Set: 16 Spectrograms")


# # Model
# Load the pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Modify the first convolutional layer to accept 1 channel instead of 3 (for grayscale images)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

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

# Number of epochs for training


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

            # Reshape images to have 1 channel (grayscale)
            images = images.unsqueeze(1)  # Now the shape is [batch_size, 1, 224, 224]

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

                images = images.unsqueeze(1)  # Now the shape is [batch_size, 1, 224, 224]
                
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
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)



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
            images = images.unsqueeze(1)
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
    images = images.unsqueeze(1)
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

