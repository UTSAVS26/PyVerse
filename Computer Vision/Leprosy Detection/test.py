import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the CustomViT model (this should match your training model architecture)
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2)  # (B, embed_dim, H'*W')
        x = x.transpose(1, 2)  # (B, H'*W', embed_dim)
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
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
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, qkv_bias, drop_rate, drop_rate)
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

def load_model(model_path, num_classes, device):
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # Check the number of classes in the saved model
    saved_num_classes = state_dict['module.head.weight'].size(0)
    
    # Initialize the model with the correct number of classes
    model = CustomViT(num_classes=saved_num_classes)
    model = nn.DataParallel(model)
    
    # Load the state dict
    model.load_state_dict(state_dict)
    
    # If the number of classes doesn't match, replace the head
    if saved_num_classes != num_classes:
        print(f"Warning: Number of classes in saved model ({saved_num_classes}) "
              f"doesn't match the specified number of classes ({num_classes}). "
              "Replacing the classification head.")
        model.module.head = nn.Linear(768, num_classes)  # Assuming embed_dim is 768
    
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, mean, std):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        _, predicted = outputs.max(1)
        probability = torch.nn.functional.softmax(outputs, dim=1)[0]
        return predicted.item(), probability[predicted.item()].item()

def display_prediction(image_path, category_id, probability, class_names):
    image = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    class_name = class_names[category_id] if class_names else f"Category {category_id}"
    plt.title(f"Predicted: {class_name}\nProbability: {probability:.2f}")
    plt.show()

def test_model(model_path, num_classes, image_paths, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_model(model_path, num_classes, device)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    for image_path in image_paths:
        try:
            image_tensor = preprocess_image(image_path, mean, std)
            category_id, probability = predict(model, image_tensor, device)
            display_prediction(image_path, category_id, probability, class_names)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

if __name__ == "__main__":
    model_path = 'best_custom_vit_mo50.pth'
    num_classes = 2  # The number of classes you expect
    
    # Specify your image paths here
    image_paths = [
        '/kaggle/input/cocoform/train/Non-lep-_210823_20_jpg.rf.507c4cfff3f2d5cd03271d4383b5cf7d.jpg',
        
    ]
    
    # Specify your class names here
    class_names = ['Leprosy','No Lep']  # Update this based on your actual classes
    
    test_model(model_path, num_classes, image_paths, class_names)