import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import random

DATA_DIR = "data/color"
MODEL_PATH = "model/plant_model.pth"
CLASSES_PATH = "model/classes.txt"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)
    class_names = full_dataset.classes

    with open(CLASSES_PATH, "w") as f:
        f.write("\n".join(class_names))

    indices = list(range(len(full_dataset)))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.85 * len(indices))
    train_ds = Subset(full_dataset, indices[:split])
    val_ds = Subset(val_dataset, indices[split:])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(8):
        model.train()
        total_loss, correct = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_correct += (model(imgs).argmax(1) == labels).sum().item()

        scheduler.step()
        print(f"epoch {epoch+1}/8 | loss: {total_loss/len(train_loader):.3f} | "
              f"train: {correct/len(train_ds)*100:.1f}% | val: {val_correct/len(val_ds)*100:.1f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print("done, model saved")