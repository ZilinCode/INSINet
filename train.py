# Example using INSINet for change detection during the training phase

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from CD_Net import INSINet  # Make sure to replace this with the actual path to your model code
from torch.utils.data import Dataset
import glob
import torch.nn.functional as F

# Step 1: Define a Custom Dataset
# This Dataset will load images and their corresponding labels for training.
class ChangeDetectionDataset(Dataset):
    def __init__(self, uppaths, transform=None):
        self.uppaths = uppaths
        self.transform = transform

    def __len__(self):
        return len(self.uppaths)

    def __getitem__(self, idx):
        uppath = self.uppaths[idx]        
        self.image_paths_A = uppath+'/A.tif'
        self.image_paths_B = uppath+'/B.tif'
        self.neighbor_paths_A = uppath+'/A_Neighbor.tif'
        self.neighbor_paths_B = uppath+'/B_Neighbor.tif'
        self.labels = uppath+'/label.tif'
        
        # Load images and labels
        img_A = Image.open(self.image_paths_A).convert('RGB')
        img_B = Image.open(self.image_paths_B).convert('RGB')
        neighbor_A = Image.open(self.neighbor_paths_A).convert('RGB')
        neighbor_B = Image.open(self.neighbor_paths_B).convert('RGB')
        label = Image.open(self.labels).convert('L')  # Grayscale label image (change/no-change mask)

        # Apply transformations
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
            neighbor_A = self.transform(neighbor_A)
            neighbor_B = self.transform(neighbor_B)
            label = transforms.ToTensor()(label)

        return img_A, img_B, neighbor_A, neighbor_B, label

# Step 2: Set Hyperparameters
# Define the training parameters such as learning rate, batch size, number of epochs, etc.
batch_size = 8
epochs = 200
learning_rate = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Step 3: Define Data Transforms
# We apply normalization and resizing to the input images. This step ensures that the images are ready for the model.
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the images to 256x256
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),  # Normalize the images
])

# Step 4: Load Dataset
dataset_path='./data/part_of_WHU/train'

# Create Dataset and DataLoader instances
uppaths=glob.glob(dataset_path+'/*')
train_dataset = ChangeDetectionDataset(uppaths, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Step 5: Initialize the Model, Loss Function, and Optimizer
# Initialize the INSINet model and move it to the selected device (GPU or CPU).
model = INSINet(in_dim=3, out_dim=2).to(device)

# Define the loss function (Binary Cross-Entropy for change detection).
criterion = torch.nn.CrossEntropyLoss()

# Define the optimizer (Adam optimizer is commonly used for deep learning models).
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step 6: Training Loop
# This loop trains the model for a specified number of epochs.
model_save_uppath='./model_parameter'
model.train()  # Set the model to training mode
for epoch in range(epochs):
    running_loss = 0.0  # To track the loss in each batch
    
    for i, (img_A, img_B, neighbor_A, neighbor_B, label) in enumerate(train_loader):
        # Move data to the device (GPU or CPU)
        img_A, img_B, neighbor_A, neighbor_B, label = img_A.to(device), img_B.to(device), neighbor_A.to(device), neighbor_B.to(device), label.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass: Compute the model output
        output = model(img_A, img_B, neighbor_A, neighbor_B)

        # Compute the loss
        label = label.squeeze(1).long() 
        loss = criterion(output, label)

        # Backward pass: Compute gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Print statistics every 10 batches
        if i % 10 == 9:
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss/10:.4f}')
            running_loss = 0.0

    # Save the model checkpoint after each epoch
    torch.save(model.state_dict(), model_save_uppath+f"/epoch_{epoch+1}.pth")

# Step 7: Save Final Model
# After all epochs are completed, save the final trained model for later use or inference.
torch.save(model.state_dict(), model_save_uppath+"/final_model.pth")
print("Finished Training and Saved the Model")

