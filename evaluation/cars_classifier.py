import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

"""simple script to train a cars classifier, can be used to extract car embeddings
    Uses a EfficientNetV2 backbone
"""

def get_cars_transform():
    transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

class CarsDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["image"].convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.ds[idx]["label"])
        return img, label

def get_dataloader(ds, batch_size=16, shuffle=True):
    wrapped = CarsDataset(ds, get_cars_transform())
    return DataLoader(wrapped, batch_size=batch_size, shuffle=shuffle, num_workers=4)


class EfficientNetEmbedder(nn.Module):
    def __init__(self, embed_dim=128, num_classes=196):
        super().__init__()
        base = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.backbone = base.features
        in_features_to_head = 1280

        self.head = nn.Sequential(
            nn.Linear(in_features_to_head, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )
        

    def forward(self, x):
        feats = self.backbone(x)  # [B, 1280, H_out, W_out]
        pooled = feats.mean([2, 3]) # [B, 1280] 
        logits = self.head(pooled)
        return logits, pooled

def main():
    ds = load_dataset("Donghyun99/Stanford-Cars")
    ds_train = ds['train']
    ds_test = ds['test']

    def train(model, dataloader, optimizer, criterion, device):
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(dataloader, desc="Training"):
            inputs = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss
    DEVICE = "cuda"
    print(f"Using device: {DEVICE}")
    model = EfficientNetEmbedder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    EPOCHS = 100
    pbar = tqdm(range(EPOCHS), total=EPOCHS, desc="Training Progress")
    for epoch in pbar:
        loss = train(model, get_dataloader(ds_train, batch_size=64), optimizer, criterion, DEVICE)
        pbar.set_postfix(loss=f"{loss:.4f}")

    MODEL_PATH = "efficientnet_cars_classifier2.pth"
    torch.save(model.state_dict(), MODEL_PATH)

    def evaluate(model, dataloader, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                inputs = images.to(device)
                labels = labels.to(device)      
                logits, _ = model(inputs)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()       
        accuracy = 100 * correct / total
        return accuracy     
    

    print("\nStarting Evaluation on Test Set...")
    test_dataloader = get_dataloader(ds_test, batch_size=64, shuffle=False)
    test_accuracy = evaluate(model, test_dataloader, DEVICE)

    print(f"Test Accuracy: {test_accuracy:.2f}%")
    

if __name__ =='__main__':
    main()

