import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms


def load_images(image_path, format):
    return Image.open(image_path).convert(format)


def list_directory(path):
    return os.listdir(path)


def join_directories(image_directory, image_names):
    return os.path.join(image_directory, image_names)


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_filenames = list_directory(image_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = join_directories(self.image_dir, self.image_filenames[idx])
        mask_path = join_directories(self.mask_dir, self.image_filenames[idx])

        image = load_images(img_path, "RGB")
        mask = load_images(mask_path, "L")

        if self.transform is None:
            self.transform = transforms.Compose([
                             transforms.Resize((512, 512)),
                             transforms.ToTensor()])

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask


def split_dataset(dataset, train_percent=0.8, val_percent=0.1):
    train_size = int(train_percent * len(dataset))
    val_size = int(val_percent * len(dataset))
    test_size = len(dataset) - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])
