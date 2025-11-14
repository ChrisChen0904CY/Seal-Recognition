import os
import random
from glob import glob
from collections import Counter
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model.model import StampClassifier
from model.simple_resnet import SimpleResNetClassifier
import warnings

warnings.filterwarnings("ignore")

# --------------------- Focal Loss ---------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        device = logits.device
        alpha = self.alpha.to(device) if self.alpha is not None else None
        ce = nn.functional.cross_entropy(logits, targets, reduction='none', weight=alpha)
        p = torch.exp(-ce)
        loss = (1 - p) ** self.gamma * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# --------------------- Dataset ---------------------
real_path = '/root/CDataset/Real'
fake_path = '/root/CDataset/Fake'
image_extensions = ['*.png', '*.jpg', '*.jpeg']

def get_image_files(directory):
    files = []
    for ext in image_extensions:
        files.extend(glob(os.path.join(directory, ext)))
    return files

real_images = get_image_files(real_path)
real_train = [f for f in real_images if 'train' in os.path.basename(f)]
real_val = [f for f in real_images if 'val' in os.path.basename(f)]
real_test = [f for f in real_images if 'test' in os.path.basename(f)]

fake_images = get_image_files(fake_path)
random.shuffle(fake_images)
n = len(fake_images)
n_train, n_val = int(0.55 * n), int(0.15 * n)
fake_train = fake_images[:n_train]
fake_val = fake_images[n_train:n_train + n_val]
fake_test = fake_images[n_train + n_val:]

train_data = [(f, 1) for f in real_train] + [(f, 0) for f in fake_train]
val_data   = [(f, 1) for f in real_val]   + [(f, 0) for f in fake_val]
test_data  = [(f, 1) for f in real_test]  + [(f, 0) for f in fake_test]

class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = ImageDataset(train_data, transform=train_transform)
val_dataset   = ImageDataset(val_data, transform=eval_transform)
test_dataset  = ImageDataset(test_data, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# --------------------- Training ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = StampClassifier().to(device)
model = SimpleResNetClassifier().to(device)

# 类别权重 alpha
train_labels = [label for _, label in train_data]
counter = Counter(train_labels)
counts = torch.tensor([counter[i] for i in range(2)], dtype=torch.float)
alpha = 1.0 / (counts + 1e-6)
alpha = alpha / alpha.sum() * len(counts) * 2
alpha = alpha.to(device)

criterion = FocalLoss(gamma=2.0, alpha=alpha)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 50
best_f1 = 0.0
best_acc = 0.0
prev_val_loss = None
early_stop_threshold = 1e-6

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

    # 验证
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_f1 = f1_score(all_labels, all_preds, average='macro')
    val_loss_avg = val_loss / val_total
    train_acc = 100.*correct/total

    print(f"\nEpoch {epoch+1}: Train Loss = {train_loss/total:.4f}, Train Acc = {train_acc:.2f}%")
    print(f"            Val Loss   = {val_loss_avg:.4f}, Val Acc   = {100.*val_correct/val_total:.2f}%, F1 = {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best.pt")
        print("✅ Saved new best model.")
    elif val_f1 == best_f1 and train_acc > best_acc:
        torch.save(model.state_dict(), "best.pt")
        print("✅ Saved new best model.")

    best_acc = max(best_acc, train_acc)
    torch.save(model.state_dict(), "last.pt")

    if prev_val_loss is not None and abs(prev_val_loss - val_loss_avg) <= early_stop_threshold:
        print("🛑 Early stopping triggered.")
        break

    prev_val_loss = val_loss_avg

# --------------------- Final Test ---------------------
model.load_state_dict(torch.load("best.pt"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
pre = precision_score(all_labels, all_preds, average='macro')
rec = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

print("📊 Test Results:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

# --------------------- External Test ---------------------
external_test_dir = '/root/seal_0'
external_images = [os.path.join(external_test_dir, f) for f in os.listdir(external_test_dir)
                   if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']]

external_preds = []
external_labels = [1] * len(external_images)

with torch.no_grad():
    for img_path in tqdm(external_images, desc="🔍 External Inference"):
        image = Image.open(img_path).convert('RGB')
        image = eval_transform(image).unsqueeze(0).to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        external_preds.append(predicted.item())

acc = accuracy_score(external_labels, external_preds)
pre = precision_score(external_labels, external_preds, average='binary')
rec = recall_score(external_labels, external_preds, average='binary')
f1 = f1_score(external_labels, external_preds, average='binary')

print("📊 External Test Results on seal_0:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")