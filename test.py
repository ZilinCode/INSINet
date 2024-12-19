import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from CD_Net import INSINet
import glob


# Step 1: Create a Custom Dataset Class to Handle Multiple Test Images
class ChangeDetectionDataset_for_Test(Dataset):
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
        
        # Load images and labels
        img_A = Image.open(self.image_paths_A).convert('RGB')
        img_B = Image.open(self.image_paths_B).convert('RGB')
        neighbor_A = Image.open(self.neighbor_paths_A).convert('RGB')
        neighbor_B = Image.open(self.neighbor_paths_B).convert('RGB')

        # Apply transformations
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
            neighbor_A = self.transform(neighbor_A)
            neighbor_B = self.transform(neighbor_B)

        return img_A, img_B, neighbor_A, neighbor_B, uppath

# Step 2: Set Hyperparameters
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Step 3: Define Image Preprocessing
# Define the necessary transformations to resize and normalize the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),  # Normalize the images
])

# Step 4: Load Your Dataset (Assuming you have a list of paths to images)
dataset_path='./data/part_of_WHU/test'

# Create a dataset object and DataLoader for batching
uppaths=glob.glob(dataset_path+'/*')
test_dataset = ChangeDetectionDataset_for_Test(uppaths, transform)
dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 5: Load the Pre-trained Model
model = INSINet(in_dim=3, out_dim=2).to(device)
model.load_state_dict(torch.load('./model_parameter/final_model.pth')) 
model.eval()

# Step 6: Process the Data in Batches
save_path = './result/'
os.makedirs(save_path, exist_ok=True)


# Loop over the data in the dataloader
with torch.no_grad():
    for batch_idx, (img_A_batch, img_B_batch, neighbor_A_batch, neighbor_B_batch, uppath_infos) in enumerate(dataloader):
        img_A_batch, img_B_batch, neighbor_A_batch, neighbor_B_batch, uppath_infos = \
            img_A_batch.to(device), img_B_batch.to(device), neighbor_A_batch.to(device), neighbor_B_batch.to(device), uppath_infos

        # Get model predictions for the current batch
        output_batch = model(img_A_batch, img_B_batch, neighbor_A_batch, neighbor_B_batch)

        # Post-process the output for each image in the batch
        for i in range(output_batch.size(0)):
            output = output_batch[i].cpu().numpy()
            output = output.argmax(axis=0)  # Assuming the output is in logits or multi-channel format

            # Save the result to a file
            output_image = Image.fromarray(output.astype('uint8'))
            outpath_name=os.path.basename(uppath_infos[i])
            output_image.save(os.path.join(save_path, f'{outpath_name}.tif'))

print("Batch processing complete. Results saved.")
