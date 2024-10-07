import os
import shutil
import zipfile
from typing import Optional

import gdown
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGE_SIZE = 150
CROP_SIZE = 100
DATASET_FLAG_FILE = "dataset_downloaded.txt"


class DogBreedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "dataset",
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        test_split: float = 0.1,
        google_drive_id: str = "1X4a5jGErxXJZ0mdNBZHhpytacEj-wCRU",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.google_drive_id = google_drive_id
        self.train_dataset = self.val_dataset = self.test_dataset = None
        self.class_names = None

    def prepare_data(self):
        # Download and extract the dataset if it doesn't exist
        if not os.path.exists(f"{self.data_dir}/{DATASET_FLAG_FILE}"):
            os.makedirs(self.data_dir, exist_ok=True)
            zip_path = os.path.join(self.data_dir, "dog_breeds.zip")

            # Download the zip file
            gdown.download(id=self.google_drive_id, output=zip_path, quiet=False)
            print("Downloaded file {zip_path}")

            # Extract the zip file
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.data_dir)

            # Create the flag file
            with open(f"{self.data_dir}/{DATASET_FLAG_FILE}", "w") as f:
                f.write("Dataset downloaded successfully")

            # Remove the zip file
            os.remove(zip_path)

            # Move contents one level up if needed
            extracted_dir = os.path.join(self.data_dir, "dataset")
            if os.path.exists(extracted_dir):
                for item in os.listdir(extracted_dir):
                    s = os.path.join(extracted_dir, item)
                    d = os.path.join(self.data_dir, item)
                    print(f"Moving {s} to {d}")
                    shutil.move(s, d)
                os.rmdir(extracted_dir)

    @property
    def normalize_transform(self):
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(CROP_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    @property
    def valid_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.CenterCrop(CROP_SIZE),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    def setup(self, stage: Optional[str] = None):
        # Ensure data is prepared
        self.prepare_data()

        # Create the full dataset
        full_dataset = ImageFolder(self.data_dir, transform=self.train_transform)

        # Store the class names
        self.class_names = full_dataset.classes

        # Calculate split sizes
        total_size = len(full_dataset)
        val_size = int(self.val_split * total_size)
        test_size = int(self.test_split * total_size)
        train_size = total_size - val_size - test_size

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        # Apply transforms to validation and test datasets
        self.val_dataset.dataset.transform = self.valid_transform
        self.test_dataset.dataset.transform = self.valid_transform

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def get_class_names(self):
        return self.class_names
