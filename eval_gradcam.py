import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # 虽然不用训练，这里保留以便后续可能扩展
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import pandas as pd  # 用于保存 Excel 文件

# ----------------------------
# 支持的图片格式列表（全部小写）
# ----------------------------
SUPPORTED_EXTS = ['.jpg', '.jpeg', '.png', '.bmp']

# ----------------------------
# 图像读取辅助函数
# ----------------------------
def open_image(path):
    if os.path.exists(path):
        return Image.open(path).convert('RGB')
    raise FileNotFoundError(f"File not found: {path}")

# ----------------------------
# 根据类别目录、鱼ID和视角返回对应图像路径（支持多种格式）
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
# 简化的 FishDataset（用于评估）
# ----------------------------
class FishDataset(Dataset):
    def __init__(self, root_dir, split="val"):
        """
        数据目录结构：
          root_dir/
              train/ 或 val/
                  class1/
                      <fishID>_side.<ext>, <fishID>_back.<ext>, <fishID>_belly.<ext>
                  class2/
                      ...
        支持的图片格式由 SUPPORTED_EXTS 指定。
        """
        self.split = split
        self.root_dir = os.path.join(root_dir, split)
        # 获取所有类别（目录名），按字母排序
        self.classes = sorted(os.listdir(self.root_dir))
        self.samples = []  # 每个元素为 (cls, fish_id)
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

        # 评估时使用 Resize + CenterCrop
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
        # 标签为类别在 self.classes 中的索引
        label = self.classes.index(cls)
        return (side_img, back_img, belly_img), label

    @property
    def fish_list(self):
        return self.samples

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
        att_weights = self.attention_fc(concat_feat)  # (batch, 3)
        alpha1 = att_weights[:, 0].unsqueeze(1)
        alpha2 = att_weights[:, 1].unsqueeze(1)
        alpha3 = att_weights[:, 2].unsqueeze(1)
        fused_feat = alpha1 * f1 + alpha2 * f2 + alpha3 * f3
        out = self.classifier(fused_feat)
        return out, att_weights  # 返回分类结果和注意力权重

# ----------------------------
# 评估及Grad-CAM可视化部分（去除训练部分）
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_dataset = FishDataset(root_dir="data", split="val")
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print(f"Validation samples: {len(val_dataset)}")

num_classes = len(val_dataset.classes)
model = MultiBranchResNet(num_classes=num_classes).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# 加载训练好的模型权重（确保best_model_attention.pth存在）
model.load_state_dict(torch.load("best_model_attention.pth", map_location=device, weights_only=True))
model.eval()

# ----------------------------
# 1. 保存分类样本（正确与错误分类）
# ----------------------------
def save_classification_samples(model, val_loader, val_dataset, device, save_dir_true="correct_samples", save_dir_false="incorrect_samples"):
    os.makedirs(save_dir_true, exist_ok=True)
    os.makedirs(save_dir_false, exist_ok=True)
    with torch.no_grad():
        for batch_idx, (images_tuple, labels) in enumerate(val_loader):
            x1, x2, x3 = images_tuple
            x1, x2, x3, labels = x1.to(device), x2.to(device), x3.to(device), labels.to(device)
            outputs, _ = model(x1, x2, x3)
            _, preds = outputs.max(1)
            for i in range(labels.size(0)):
                idx_in_dataset = batch_idx * val_loader.batch_size + i
                cls, fish_id = val_dataset.fish_list[idx_in_dataset]
                side_path = get_image_path(os.path.join("data", "val", cls), fish_id, "side")
                back_path = get_image_path(os.path.join("data", "val", cls), fish_id, "back")
                belly_path = get_image_path(os.path.join("data", "val", cls), fish_id, "belly")
                if side_path is None or back_path is None or belly_path is None:
                    continue
                side_img = open_image(side_path)
                back_img = open_image(back_path)
                belly_img = open_image(belly_path)
                save_dir_now = save_dir_true if preds[i] == labels[i] else save_dir_false
                side_img.save(os.path.join(save_dir_now, os.path.basename(side_path)))
                back_img.save(os.path.join(save_dir_now, os.path.basename(back_path)))
                belly_img.save(os.path.join(save_dir_now, os.path.basename(belly_path)))
    print(f"Correct samples saved in '{save_dir_true}', incorrect samples saved in '{save_dir_false}'.")

save_classification_samples(model, val_loader, val_dataset, device, save_dir_true="correct_samples", save_dir_false="incorrect_samples")

# ----------------------------
# 2. Grad-CAM 可视化函数
# ----------------------------
def get_model_input_and_original(fish_entry, view, root_dir="data", split="val", input_size=(256,256)):
    cls, fish_id = fish_entry
    cls_dir = os.path.join(root_dir, split, cls)
    path = get_image_path(cls_dir, fish_id, view)
    if path is None:
        raise FileNotFoundError(f"Image for {fish_id} with view {view} not found.")
    original_pil = open_image(path)
    transform_input = transforms.Compose([
         transforms.Resize(input_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])
    ])
    model_input = transform_input(original_pil)
    return model_input, original_pil

class GradCAM:
    def __init__(self, model_branch, target_layer):
        self.model_branch = model_branch
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)
    
    def forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()
    
    def generate_cam(self, input_tensor, class_idx=None):
        output = self.model_branch(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[0, class_idx]
        self.model_branch.zero_grad()
        score.backward()
        alpha = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (alpha * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = nn.functional.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam[0,0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def save_gradcam_visualization_for_view(model, fish_entry, view, device, save_dir="Grad-CAM", filename_prefix="sample", input_size=(256,256)):
    os.makedirs(save_dir, exist_ok=True)
    input_tensor, original_pil = get_model_input_and_original(fish_entry, view, root_dir="data", split="val", input_size=input_size)
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    if view == "side":
        branch = model.module.branch1 if isinstance(model, nn.DataParallel) else model.branch1
    elif view == "back":
        branch = model.module.branch2 if isinstance(model, nn.DataParallel) else model.branch2
    elif view == "belly":
        branch = model.module.branch3 if isinstance(model, nn.DataParallel) else model.branch3
    
    # 选用 layer4 的最后一个 block（避免 AdaptiveAvgPool2d）
    target_layer = branch[-2][-1]
    
    gradcam = GradCAM(model_branch=branch, target_layer=target_layer)
    cam = gradcam.generate_cam(input_tensor)
    
    original_np = np.array(original_pil).astype(np.float32) / 255.0
    cam_up = cv2.resize(cam, (original_np.shape[1], original_np.shape[0]))
    heatmap = cv2.applyColorMap((cam_up*255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = 0.5 * original_np + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    cv2.imwrite(os.path.join(save_dir, f"{filename_prefix}_{view}_original.png"), (original_np*255).astype(np.uint8))
    cv2.imwrite(os.path.join(save_dir, f"{filename_prefix}_{view}_gradcam_heatmap.png"), (heatmap*255).astype(np.uint8))
    cv2.imwrite(os.path.join(save_dir, f"{filename_prefix}_{view}_overlay.png"), (overlay*255).astype(np.uint8))
    print(f"Grad-CAM for {view} saved as {filename_prefix}_{view} images.")
