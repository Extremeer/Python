import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# 定义超分辨率模型
class SuperResolutionModel(nn.Module):
    def __init__(self, upscale_factor=2, num_channels=3):
        super(SuperResolutionModel, self).__init__()
        # 定义模型的层，这里只是一个简单示例，请根据任务设计更复杂的模型
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, num_channels * upscale_factor ** 2, kernel_size=3, padding=1)
        self.upsample = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.upsample(x)
        return x

# 定义自定义数据集类，包含图像对
class CustomDataset(Dataset):
    def __init__(self, data_folder, upscale_factor=2, transform=None):
        self.data_folder = data_folder
        self.upscale_factor = upscale_factor
        self.image_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(('.png', '.jpg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")

        # 创建低分辨率图像（模拟输入）
        low_res_transform = transforms.Compose([
            transforms.Resize((image.size[1] // self.upscale_factor, image.size[0] // self.upscale_factor)),
            #transforms.Resize((image.size[1], image.size[0])),
        ])
        low_res_image = low_res_transform(image)

        if self.transform:
            low_res_image = self.transform(low_res_image)
            image = self.transform(image)

        #print(image.size[0])

        return low_res_image, image  # 返回低分辨率图像和高分辨率图像对

# 数据预处理和加载器
data_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = CustomDataset("SuperResolution/dataset/train", transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 创建模型实例并将其移到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SuperResolutionModel().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, targets = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item()}")

# 保存模型
torch.save(model.state_dict(), "SuperResolution/super_resolution_model.pth")