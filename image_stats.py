import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm


train_dir = "/media/iot/HDD2TB/eyepac-light-v2-512-jpg/train"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
train_dataset_for_stats = ImageFolder(train_dir, transform=transform)
train_loader_for_stats = DataLoader(train_dataset_for_stats, batch_size=8, shuffle=True, num_workers=4)

mean = 0.0
std = 0.0
nb_samples = 0.0
for data, _ in tqdm(train_loader_for_stats, desc="Calculating Stats ..."):
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples
print("Mean:", mean)
print("Std:", std)

# Output:
# Mean: tensor([0.3569, 0.2274, 0.1467])
# Std: tensor([0.2309, 0.1543, 0.1033])