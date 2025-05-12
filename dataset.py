import os
from PIL import Image
from torch.utils.data import Dataset

class FacialExpressionDataset(Dataset):
    """
    Custom Dataset class for loading facial expression images.
    Assumes data_dir contains subdirectories, each named after an expression
    (e.g., 'happy', 'sad'), and each subdirectory contains images for that expression.
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Path to the main data directory (e.g., '/aml-fer/data').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Discover classes (expression names) and map them to integer labels
        # Sorting ensures consistent class_to_idx mapping
        class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

        if not class_names:
            raise FileNotFoundError(f"No subdirectories found in {data_dir}. "
                                    f"Expected subdirectories for each expression class.")

        for i, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = i
            self.idx_to_class[i] = class_name

        # Load image paths and their corresponding labels
        for class_name, idx in self.class_to_idx.items():
            class_path = os.path.join(data_dir, class_name)
            for img_filename in os.listdir(class_path):
                img_path = os.path.join(class_path, img_filename)
                # Ensure it's a file and has a common image extension
                if os.path.isfile(img_path) and \
                   img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {data_dir} subdirectories. "
                                    f"Please check the directory structure and image files.")

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Fetches the image and its label at the given index.
        Args:
            idx (int): Index of the sample to fetch.
        Returns:
            tuple: (image, label) where image is the transformed image
                   and label is the integer label of the expression.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image using Pillow library
        image = Image.open(img_path).convert('RGB')  # Convert to RGB to ensure 3 channels

        if self.transform:
            image = self.transform(image)

        return image, label
