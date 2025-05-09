import os
from PIL import Image
import torch
# import torch_npu
# from torch_npu.contrib import transfer_to_npu
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from logging import getLogger
from collections import OrderedDict

logger = getLogger()

def make_imagenet2012(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    index_targets=False,
    naflex=False,
):
    # dataset = ImageNetDataset(
    #     image_folder=image_folder,
    #     transform=transform,
    #     naflex=naflex,
    #     index_targets=index_targets)
    dataset = datasets.ImageFolder(
        image_folder,
        transform=transform,
        
    )
    logger.info('ImageNet dataset created')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('ImageNet unsupervised data loader created')

    return dataset, data_loader, None

class ImageNetDataset(Dataset):
    def __init__(self, image_folder, naflex=False, transform=None, index_targets=False):
        self.image_folder = image_folder
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names  = OrderedDict()
        self.naflex = naflex
        print(index_targets)

        assert os.path.isfile(index_targets)
        with open(index_targets, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = parts[0]
                label = int(parts[1]) - 1 # the index_targets start from 1
                class_name = [' '.join(parts[2:]).replace("_", " ")]
                self.class_names[class_id] = (label, class_name)
        
        for class_dir in os.listdir(image_folder):
            class_path = os.path.join(image_folder, class_dir)
            if os.path.isdir(class_path):
                label, class_name = self.class_names[class_dir]
                for img_file in os.listdir(class_path):
                    if img_file.endswith('.JPEG') or img_file.endswith('.jpeg'):
                        img_path = os.path.join(class_path, img_file)
                        self.images.append(img_path)
                        self.labels.append((label, class_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label, class_name = self.labels[idx]

        if self.naflex:
            if self.transform:
                image_input = self.transform(image)

            image_tensor = image_input["pixel_values"]
            pixel_attention_mask = image_input["pixel_attention_mask"]
            spatial_shapes = image_input["spatial_shapes"]

            return image_tensor, pixel_attention_mask, spatial_shapes, label

        else:
            if self.transform:
                image = self.transform(image)

            return image, label
