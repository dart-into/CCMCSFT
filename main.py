import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torchvision.models as models
import vit
import cv2
import numpy as np
import pandas as pd
import Resnet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 读取CSV文件
dataframe = pd.read_csv('/home/user/nihao/V1_fan.csv', encoding='GBK')


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]

        # Load and preprocess image
        original_image = Image.open(img_name)

        image_hsv = self.load_and_preprocess_image_HSV(original_image)

        image_lab = self.load_and_preprocess_image_LAB(original_image)

        # Split HSV channels
        image_h, image_s, image_v = cv2.split(image_hsv)

        # Split LAB channels
        image_l, image_a, image_b = cv2.split(image_lab)

        image_hsl = np.stack((image_h, image_s, image_l), axis=-1)

        # Convert numpy array to PIL image
        pil_image = Image.fromarray(image_lab)

        if self.transform:
            image1 = self.transform(original_image)
            image2 = self.transform(pil_image)

        label = self.dataframe.iloc[idx, 1]

        return image1, image2, label

    def load_and_preprocess_image_HSV(self, image):
        # Convert image from PIL format to numpy array
        image_np = np.array(image)
        # Convert image from RGB to HSV
        image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        return image_hsv

    def load_and_preprocess_image_LAB(self, image):
        # Convert image from PIL format to numpy array
        image_np = np.array(image)
        # Convert image from RGB to LAB
        image_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        return image_lab


# 定义训练数据预处理转换
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为相同的尺寸
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 定义测试数据预处理转换
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为相同的尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 创建自定义数据集
train_dataset = CustomDataset(dataframe=dataframe, transform=train_transform)
test_dataset = CustomDataset(dataframe=dataframe, transform=test_transform)

# 划分数据集，80% 训练集，20% 测试集
train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)

pretrained_cfg_overlay = {'file': r"/home/user/nihao/pytorch_model.bin"}
vit_model = timm.create_model('vit_base_patch16_224', pretrained_cfg_overlay=pretrained_cfg_overlay,
                              pretrained=True)
VIT = vit.VisionTransformer()
state_dict = vit_model.state_dict()
VIT.load_state_dict(state_dict)

resnet1 = models.resnet50(pretrained = True)
net_re1 = Resnet.resnet50(pretrained=False)
state_dict1 = resnet1.state_dict()
net_re1.load_state_dict(state_dict1)

resnet2 = models.resnet50(pretrained = True)
net_re2 = Resnet.resnet50(pretrained=False)
state_dict2 = resnet2.state_dict()
net_re2.load_state_dict(state_dict2)



# 定义自定义的ResNet_18模型
class ResNet_18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_18, self).__init__()


        # 加载预训练的ResNet-18模型
        self.resnet1 = net_re1
        self.resnet2 = net_re2
        
        self.conv1 = nn.Conv2d(2048, 768, 3, 1, 1)
        self.conv2 = nn.Conv2d(1024, 768, 3, 1, 1)
        self.conv3 = nn.Conv2d(512, 768, 3, 1, 1)
        self.conv4 = nn.Conv2d(2304, 768, 3, 1, 1)
        
        self.maxpool3 = nn.MaxPool2d(2)
        self.maxpool4 = nn.MaxPool2d(4)
        
        self.vit1 = VIT
        self.vit2 = VIT
        self.vit3 = VIT
        
        self.fc1 = nn.Linear(3000, num_classes)

        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x1, x2):
    
        feature1_rgb,feature2_rgb,feature3_rgb = self.resnet1(x1)
        feature1_lab,feature2_lab,feature3_lab = self.resnet2(x2)
        
        out_1024 = torch.concat([feature3_rgb, feature3_lab], dim=1)
        out_512 = torch.concat([feature2_rgb, feature2_lab], dim=1)
        out_256 = torch.concat([feature1_rgb, feature1_lab], dim=1)
        
        out1 = self.conv1(out_1024)
        out2 = self.conv2(out_512)
        out3 = self.conv3(out_256)
        
        out2 = self.maxpool3(out2)
        out3 = self.maxpool4(out3)
        
        out1 = self.vit1(out1)
        out2 = self.vit2(out2)
        out3 = self.vit3(out3)
        
        out = torch.concat([out1, out2, out3], dim=1)

        out = self.fc1(out)

        return out


# 定义模型
model = ResNet_18(num_classes=18)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for image1, image2, labels in tqdm(train_loader):
        image1, image2, labels = image1.to(device), image2.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(image1, image2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

# 在训练完成后进行测试集验证
model.eval()
val_loss = 0.0
correct = 0
total = 0
all_labels = []
all_preds = []

with torch.no_grad():
    for image1, image2, labels in test_loader:
        image1, image2, labels = image1.to(device), image2.to(device), labels.to(device)
        outputs = model(image1, image2)
        loss = criterion(outputs, labels)

        val_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 保存真实标签和预测标签
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

val_loss = val_loss / len(test_loader)
val_accuracy = 100 * correct / total

# 打印最终的验证结果
print(f"Final Test Results - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# 计算分类报告
report = classification_report(all_labels, all_preds, target_names=[str(i).zfill(4) for i in range(18)], output_dict=True)

# 打印每个类别的精确率、召回率、F1 分数，格式化为4位有效数字
for class_label, metrics in report.items():
    if class_label.isdigit():  # 仅打印类别的准确率等
        print(f"Class {class_label}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1-score']:.4f}")
        print(f"  Support:   {metrics['support']}")