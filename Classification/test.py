import os
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model.model import StampClassifier
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Define the path to the external test set
external_test_dir = '/root/seal_0'

# Define image transformation (same as validation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StampClassifier().to(device)
model.load_state_dict(torch.load("best.pt", map_location=device))
model.eval()

# Collect all image paths from the external test directory
image_extensions = ['.png', '.jpg', '.jpeg']
image_paths = [os.path.join(external_test_dir, fname) for fname in os.listdir(external_test_dir)
               if os.path.splitext(fname)[1].lower() in image_extensions]

# Prepare ground truth labels (all 1s)
true_labels = [1] * len(image_paths)
pred_labels = []

# Run inference
with torch.no_grad():
    for img_path in tqdm(image_paths, desc="🔍 Inference on seal_0"):
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        pred_labels.append(predicted.item())


# Compute metrics
acc = accuracy_score(true_labels, pred_labels)
pre = precision_score(true_labels, pred_labels, average='binary')
rec = recall_score(true_labels, pred_labels, average='binary')
f1 = f1_score(true_labels, pred_labels, average='binary')

# Print results
print("📊 External Test Results on seal_0:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
