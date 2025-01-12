import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class CustomViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=2, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x

class LeprosyDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.images = self.annotations['images']
        self.categories = {0: "Leprosy", 1: "Non Leprosy"}
        self.img_to_label = {}
        
        for ann in self.annotations['annotations']:
            original_category = ann['category_id']
            binary_label = 0 if original_category in [0, 1] else 1
            self.img_to_label[ann['image_id']] = binary_label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.img_to_label[img_info['id']]
        return image, label

def calculate_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1': f1_score(y_true, y_pred, average='binary'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'confusion_matrix_{timestamp}.png')
    plt.close()

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
    
    metrics = calculate_metrics(all_labels, all_predictions)
    return running_loss / len(loader), metrics

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    metrics = calculate_metrics(all_labels, all_predictions)
    return running_loss / len(loader), metrics

def main():
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets and dataloaders
    train_dataset = LeprosyDataset('/kaggle/input/cocoform/train', 
                                 '/kaggle/input/cocoform/train/_annotations.coco.json', 
                                 transform=transform)
    val_dataset = LeprosyDataset('/kaggle/input/cocoform/valid', 
                               '/kaggle/input/cocoform/valid/_annotations.coco.json', 
                               transform=transform)
    test_dataset = LeprosyDataset('/kaggle/input/cocoform/test', 
                                '/kaggle/input/cocoform/test/_annotations.coco.json', 
                                transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize model and move to device
    model = CustomViT(num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model = model.to(device)

    # Set up loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop with detailed metrics
    num_epochs = 20
    best_val_acc = 0
    all_metrics = []

    print("Training Started...")
    print("-" * 120)
    print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Train Acc':^9} | {'Train Prec':^10} | {'Train Rec':^9} | {'Train F1':^8} | "
          f"{'Val Loss':^8} | {'Val Acc':^7} | {'Val Prec':^8} | {'Val Rec':^7} | {'Val F1':^6}")
    print("-" * 120)

    for epoch in range(num_epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Store metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_metrics': train_metrics,
            'val_loss': val_loss,
            'val_metrics': val_metrics
        }
        all_metrics.append(epoch_metrics)
        
        # Print metrics
        print(f"{epoch+1:6d} | {train_loss:10.4f} | {train_metrics['accuracy']:9.4f} | "
              f"{train_metrics['precision']:10.4f} | {train_metrics['recall']:9.4f} | {train_metrics['f1']:8.4f} | "
              f"{val_loss:8.4f} | {val_metrics['accuracy']:7.4f} | {val_metrics['precision']:8.4f} | "
              f"{val_metrics['recall']:7.4f} | {val_metrics['f1']:6.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), 'best_custom_vit_mo.pth')
            print("✓ New best model saved!")

        # Plot confusion matrix for this epoch
        plot_confusion_matrix(train_metrics['confusion_matrix'], 
                            classes=["Leprosy", "Non-Leprosy"],
                            title=f'Training Confusion Matrix - Epoch {epoch+1}')

    print("-" * 120)
    print("Training Complete!")

    # Test evaluation
    print("\nEvaluating Best Model on Test Set...")
    model.load_state_dict(torch.load('best_custom_vit_mo.pth'))
    test_loss, test_metrics = validate(model, test_loader, criterion, device)

    print("\nFinal Test Results:")
    print("-" * 50)
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1-Score:  {test_metrics['f1']:.4f}")
    print("-" * 50)

    # Plot final test confusion matrix
    plot_confusion_matrix(test_metrics['confusion_matrix'], 
                        classes=["Leprosy", "Non-Leprosy"],
                        title='Final Test Confusion Matrix')

    # Save all metrics to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'training_metrics_{timestamp}.txt', 'w') as f:
        f.write("Training Metrics Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Write epoch-wise metrics
        f.write("Epoch-wise Metrics:\n")
        f.write("-" * 50 + "\n")
        for metric in all_metrics:
            f.write(f"Epoch {metric['epoch']}:\n")
            f.write(f"  Training:\n")
            f.write(f"    Loss: {metric['train_loss']:.4f}\n")
            f.write(f"    Accuracy: {metric['train_metrics']['accuracy']:.4f}\n")
            f.write(f"    Precision: {metric['train_metrics']['precision']:.4f}\n")
            f.write(f"    Recall: {metric['train_metrics']['recall']:.4f}\n")
            f.write(f"    F1-Score: {metric['train_metrics']['f1']:.4f}\n")
            f.write(f"  Validation:\n")
            f.write(f"    Loss: {metric['val_loss']:.4f}\n")
            f.write(f"    Accuracy: {metric['val_metrics']['accuracy']:.4f}\n")
            f.write(f"    Precision: {metric['val_metrics']['precision']:.4f}\n")
            f.write(f"    Recall: {metric['val_metrics']['recall']:.4f}\n")
            f.write(f"    F1-Score: {metric['val_metrics']['f1']:.4f}\n")
            f.write("\n")
        
        # Write final test metrics
        f.write("\nFinal Test Metrics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Loss: {test_loss:.4f}\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {test_metrics['f1']:.4f}\n")

    # Plot training history
    plt.figure(figsize=(12, 8))
    epochs = range(1, num_epochs + 1)
    
    # Plot training metrics
    plt.subplot(2, 2, 1)
    plt.plot([m['train_loss'] for m in all_metrics], label='Train Loss')
    plt.plot([m['val_loss'] for m in all_metrics], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot([m['train_metrics']['accuracy'] for m in all_metrics], label='Train Accuracy')
    plt.plot([m['val_metrics']['accuracy'] for m in all_metrics], label='Val Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot([m['train_metrics']['precision'] for m in all_metrics], label='Train Precision')
    plt.plot([m['train_metrics']['recall'] for m in all_metrics], label='Train Recall')
    plt.plot([m['train_metrics']['f1'] for m in all_metrics], label='Train F1')
    plt.title('Training Metrics History')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot([m['val_metrics']['precision'] for m in all_metrics], label='Val Precision')
    plt.plot([m['val_metrics']['recall'] for m in all_metrics], label='Val Recall')
    plt.plot([m['val_metrics']['f1'] for m in all_metrics], label='Val F1')
    plt.title('Validation Metrics History')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_history_{timestamp}.png')
    plt.close()

if __name__ == "__main__":
    main()