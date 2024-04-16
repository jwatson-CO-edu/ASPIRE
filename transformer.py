import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import timm
from PIL import Image
import numpy as np

class RobotPoseDataset(Dataset):
    def __init__(self, image_dir, transform = None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted([file for file in os.listdir(image_dir) if file.startswith('color_image_') and file.endswith('.png')])
        self.pose_files = sorted([file for file in os.listdir(image_dir) if file.startswith('x_pose_') and file.endswith('.txt')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        
        pose_file = self.pose_files[idx]
        pose_path = os.path.join(self.image_dir, pose_file)
        
        with open(pose_path, 'r') as file:
            data_string = file.read()
            pose = np.fromstring(data_string.strip('[]'), sep=',', dtype = np.float32)

        if self.transform:
            image = self.transform(image) # apply transformation, conversion to tensor

        pose = torch.from_numpy(pose) # convert numpy array to tensor

        return image, pose, idx

class PosePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # load pre-trained ViT
        self.vit = timm.create_model('vit_base_patch16_224', pretrained = True, num_classes = 0) # remove classification head
        self.regressor = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 6) # x, y, z, alpha, beta, gamma
        )

    def forward(self, x):
        features = self.vit(x) # features, vision transformer
        poses = self.regressor(features)
        return poses

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
])

dataset = RobotPoseDataset(image_dir = 'data/', transform = transform)
train_size = int(0.8 * len(dataset))
val_test_size = len(dataset) - train_size
val_size = test_size = val_test_size // 2
train_dataset, val_test_dataset = random_split(dataset, [train_size, val_test_size])
val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size = 4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size = 4, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PosePredictor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.MSELoss()

def test(model, test_dataloader):
    model.eval()
    with torch.no_grad():
        for images, poses, idxs in test_dataloader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.cpu().numpy()
            print(f'Image Number: {idxs.item()}, Predicted Pose: {outputs}')

def train(model, train_dataloader, val_dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for images, poses, _ in train_dataloader:  # ignore idxs during training
            images, poses = images.to(device), poses.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, poses)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, poses, _ in val_dataloader:  # ignore idxs during validation
                images, poses = images.to(device), poses.to(device)
                outputs = model(images)
                loss = criterion(outputs, poses)
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(train_dataloader)}, Val Loss: {val_loss / len(val_dataloader)}')

train(model, train_dataloader, val_dataloader, optimizer, criterion, epochs = 10)
test(model, test_dataloader)  # run the test after training is complete
