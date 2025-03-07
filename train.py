import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# ----------------------------
# ----------------------------
SUPPORTED_EXTS = ['.jpg', '.jpeg', '.png', '.bmp']

# ----------------------------
# ----------------------------
def get_image_path(cls_dir, fish_id, view):
    """
    尝试查找 fish_id 对应的指定视角（side/back/belly）图片，
    支持 SUPPORTED_EXTS 中的任一格式，返回第一个存在的路径，否则返回 None
    """
    for ext in SUPPORTED_EXTS:
        path = os.path.join(cls_dir, f"{fish_id}_{view}{ext}")
        if os.path.exists(path):
            return path
    return None

# ----------------------------
# A simplified FishDataset
# ----------------------------
class FishDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        """
        Data directory structure：
        root_dir/
            train/ or val/
                class1/
                    <fishID>_side.<ext>, <fishID>_back.<ext>, <fishID>_belly.<ext>
                class2/
                    ...
        The supported image formats are specified by SUPPORTED_EXTS.
        """
        self.split = split
        self.root_dir = os.path.join(root_dir, split)
        # Get all categories (catalog names), sorted alphabetically
        self.classes = sorted(os.listdir(self.root_dir))
        self.samples = []
        # Iterate through each category catalog to find a sample that meets the requirements (must contain side, back, and belly perspectives)
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for fname in os.listdir(cls_dir):
                lower_fname = fname.lower()
                for ext in SUPPORTED_EXTS:
                    suffix = f"_side{ext}"
                    if lower_fname.endswith(suffix):
                        fish_id = fname[:-len(suffix)]
                        side_path = get_image_path(cls_dir, fish_id, "side")
                        back_path = get_image_path(cls_dir, fish_id, "back")
                        belly_path = get_image_path(cls_dir, fish_id, "belly")
                        if side_path and back_path and belly_path:
                            self.samples.append((cls, fish_id))
                        else:
                            print(f"Skipping {fish_id} in class {cls}: missing one or more views.")
                        break

        # Define image preprocessing/data enhancement
        if split == "train":
            self.transform = transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        cls, fish_id = self.samples[idx]
        cls_dir = os.path.join(self.root_dir, cls)
        side_path = get_image_path(cls_dir, fish_id, "side")
        back_path = get_image_path(cls_dir, fish_id, "back")
        belly_path = get_image_path(cls_dir, fish_id, "belly")
        side_img = self.transform(Image.open(side_path).convert('RGB'))
        back_img = self.transform(Image.open(back_path).convert('RGB'))
        belly_img = self.transform(Image.open(belly_path).convert('RGB'))
        # 根据类别在 self.classes 中的索引作为标签
        label = self.classes.index(cls)
        return (side_img, back_img, belly_img), label

# ----------------------------
# Multi-branch ResNet with Attention-based Feature Fusion
# ----------------------------
class MultiBranchResNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiBranchResNet, self).__init__()
        base_model_1 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        base_model_2 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        base_model_3 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.branch1 = nn.Sequential(*list(base_model_1.children())[:-1])
        self.branch2 = nn.Sequential(*list(base_model_2.children())[:-1])
        self.branch3 = nn.Sequential(*list(base_model_3.children())[:-1])
        self.feature_dim = base_model_1.fc.in_features
        
        # 注意力模块：输入3*feature_dim，输出3个权重，并归一化（Softmax）
        self.attention_fc = nn.Sequential(
            nn.Linear(self.feature_dim * 3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)
    
    def forward(self, x1, x2, x3):
        f1 = self.branch1(x1).view(x1.size(0), -1)
        f2 = self.branch2(x2).view(x2.size(0), -1)
        f3 = self.branch3(x3).view(x3.size(0), -1)
        concat_feat = torch.cat([f1, f2, f3], dim=1)
        att_weights = self.attention_fc(concat_feat)  # 形状：(batch, 3)
        alpha1 = att_weights[:, 0].unsqueeze(1)
        alpha2 = att_weights[:, 1].unsqueeze(1)
        alpha3 = att_weights[:, 2].unsqueeze(1)
        fused_feat = alpha1 * f1 + alpha2 * f2 + alpha3 * f3
        out = self.classifier(fused_feat)
        return out, att_weights  # 返回分类输出及注意力权重

# ----------------------------
# Training, Validation, and Evaluation
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = FishDataset(root_dir="data", split="train")
val_dataset = FishDataset(root_dir="data", split="val")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# 类别数由数据集中的类别决定
num_classes = len(train_dataset.classes)
model = MultiBranchResNet(num_classes=num_classes).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10
best_val_acc = 0.0
save_path = "best_model_attention.pth"

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
log_lines = []  # 保存每个 epoch 的日志

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images_tuple, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        x1, x2, x3 = images_tuple
        x1, x2, x3, labels = x1.to(device), x2.to(device), x3.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, att_weights = model(x1, x2, x3)  # 解包返回值
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    train_loss = total_loss / total
    train_acc = correct / total
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images_tuple, labels in val_loader:
            x1, x2, x3 = images_tuple
            x1, x2, x3, labels = x1.to(device), x2.to(device), x3.to(device), labels.to(device)
            outputs, _ = model(x1, x2, x3)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss = total_loss / total
    val_acc = correct / total
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    
    log_line = f"Epoch {epoch+1}/{num_epochs}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}"
    print(log_line)
    log_lines.append(log_line)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print("Best model saved!")

# 保存训练日志到文件
with open("training_log.txt", "w") as f:
    for line in log_lines:
        f.write(line + "\n")

print("Training completed!")

# 绘制并保存训练曲线
epochs_range = range(1, num_epochs+1)
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(epochs_range, train_loss_list, 'o-', label="Train Loss")
plt.plot(epochs_range, val_loss_list, 's-', label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(epochs_range, train_acc_list, 'o-', label="Train Acc")
plt.plot(epochs_range, val_acc_list, 's-', label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("training_curves.png", dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------
# Evaluation: Confusion Matrix and Classification Report
# ----------------------------
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images_tuple, labels in val_loader:
        x1, x2, x3 = images_tuple
        x1, x2, x3, labels = x1.to(device), x2.to(device), x3.to(device), labels.to(device)
        outputs, _ = model(x1, x2, x3)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=val_dataset.classes)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
np.savetxt("confusion_matrix.txt", cm, fmt="%d", header="Confusion Matrix")
with open("classification_report.txt", "w") as f:
    f.write(report)
print("Evaluation results saved!")

# ----------------------------
# Attention weights statistics and visualization
# ----------------------------
att_weights_list = []
labels_list = []
model.eval()
with torch.no_grad():
    for images_tuple, labels in tqdm(val_loader, desc="Collecting attention weights"):
        x1, x2, x3 = images_tuple
        x1, x2, x3, labels = x1.to(device), x2.to(device), x3.to(device), labels.to(device)
        _, att_weights = model(x1, x2, x3)
        att_weights_list.append(att_weights.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
att_weights_all = np.concatenate(att_weights_list, axis=0)  # shape: (N, 3)
labels_all = np.concatenate(labels_list, axis=0)            # shape: (N,)

# 保存注意力权重到 Excel 文件
import pandas as pd
df = pd.DataFrame(att_weights_all, columns=['alpha_side', 'alpha_back', 'alpha_belly'])
df['class'] = [val_dataset.classes[i] for i in labels_all]
df.to_excel("attention_weights.xlsx", index=False)
print("Attention weights saved to attention_weights.xlsx")

# 绘制箱型图（高分辨率）
plt.figure(figsize=(8,6))
plt.boxplot([att_weights_all[:,0], att_weights_all[:,1], att_weights_all[:,2]],
            labels=['side', 'back', 'belly'], patch_artist=True)
plt.xlabel("View", fontsize=14)
plt.ylabel("Attention Weight", fontsize=14)
plt.title("Distribution of Attention Weights", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("attention_weights_boxplot.png", dpi=300)
plt.show()

# 绘制每个类别的平均注意力权重条形图
mean_weights_per_class = {}
for cls in val_dataset.classes:
    indices = [i for i, lbl in enumerate(labels_all) if val_dataset.classes[lbl] == cls]
    if indices:
        mean_weights = att_weights_all[indices].mean(axis=0)
        mean_weights_per_class[cls] = mean_weights

classes = list(mean_weights_per_class.keys())
mean_side = [mean_weights_per_class[cls][0] for cls in classes]
mean_back = [mean_weights_per_class[cls][1] for cls in classes]
mean_belly = [mean_weights_per_class[cls][2] for cls in classes]
x = np.arange(len(classes))
width = 0.25
plt.figure(figsize=(10,6))
plt.bar(x - width, mean_side, width, label='side')
plt.bar(x, mean_back, width, label='back')
plt.bar(x + width, mean_belly, width, label='belly')
plt.xlabel("Class", fontsize=14)
plt.ylabel("Mean Attention Weight", fontsize=14)
plt.title("Mean Attention Weights per Class", fontsize=16)
plt.xticks(x, classes, rotation=45)
plt.legend(fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("attention_weights_barplot.png", dpi=300)
plt.show()
