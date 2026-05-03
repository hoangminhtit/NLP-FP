from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from config import config

DATASET_NAME = getattr(config, "DATASET_NAME", "flaviagiammarino/path-vqa")

image_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

class VQA_dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]['image'].convert('RGB')
        question = self.data[idx]['question']
        answer = self.data[idx]['answer']
        if self.transform:
            image = self.transform(image)
        return image, question, answer


def build_dataloaders(dataset_name=DATASET_NAME, batch_size=None):
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    dataset = load_dataset(dataset_name)
    train_data = dataset['train']
    val_data = dataset['validation']
    test_data = dataset['test']

    train_dataset = VQA_dataset(train_data, transform=image_transforms)
    val_dataset = VQA_dataset(val_data, transform=image_transforms)
    test_dataset = VQA_dataset(test_data, transform=image_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = build_dataloaders()

if __name__ == "__main__":
    train_loader, _, _ = build_dataloaders()
    train_dataset = train_loader.dataset
    random_indices = np.random.choice(len(train_dataset), 3)
    for idx in random_indices:
        idx = int(idx)  # Chuyển đổi chỉ số thành kiểu int
        image, question, answer = train_dataset[idx]
        image = image.permute(1, 2, 0).numpy()

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(f"Question: {question} \n Answer: {answer}", fontsize=16)
        plt.axis('off')
        plt.show()
    print(type(train_dataset[0][1]))