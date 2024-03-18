from torchvision import transforms
from PIL import Image
import os
import cv2
import torch
from config import *
import pandas as pd

stats = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, captions, tokenizer, transforms, rows: int = None):
        if isinstance(image_paths, str):
            image_paths = os.listdir(image_paths)
        self.image_paths = list(image_paths)
        self.captions = list(captions)
        if rows:
            self.captions = self.captions[:rows]
            self.image_paths = self.image_paths[:rows]
        self.encoded_captions = tokenizer(self.captions, padding=True, truncation=True, max_length=200, return_tensors='pt')
        self.transforms = transforms

    def __getitem__(self, idx):
        image_path = os.path.join("images", self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        return {'image': image,  # .unsqueeze(0),
                'caption': self.captions[idx],
                'input_ids': self.encoded_captions['input_ids'][idx],  # .unsqueeze(0),
                'attention_mask': self.encoded_captions['attention_mask'][idx]}  # .unsqueeze(0)}

    def __len__(self):
        return len(self.captions)

def create_dataloader(ds: torch.utils.data.Dataset, batch_size: int = batch_size):
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=workers, shuffle=True)

def create_dataframe(data_file: str = "data.csv", rows: int = None):
    df = pd.read_csv("data.csv")
    df = df.dropna()
    print("images: ", len(os.listdir("images/")))
    df = df.merge(pd.DataFrame({"image_name": os.listdir("images/")}), on="image_name", how="inner")
    if rows:
        df = df[:rows]
    print("captions: ", len(df))
    return df

def denorm(img_tensors):
    "Denormalize image tensor with specified mean and std"
    return img_tensors * stats[1][0] + stats[0][0]


image_transform = transforms.Compose([transforms.Resize(image_size),
                                      transforms.CenterCrop(image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(*stats)
                                      ])

print("g")
