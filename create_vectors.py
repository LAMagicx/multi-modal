import torch
import torchvision
import clip
from PIL import Image
import os
import skimage
import IPython.display
import matplotlib.pyplot as plt
import numpy as np
from data import create_dataframe, create_dataloader, Dataset, image_transform
from retrival import CustImgRetrieval as ImgRetrieval
from model import Model
from config import model_name


print("Loading model")
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP model
# model, preprocess = clip.load("ViT-B/32", device=device)

model = Model()
model.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))

ret = ImgRetrieval(image_paths=os.listdir("images")[:10000], model=model, process_image=image_transform, data_file="custom_image_vectors.pt")


df = create_dataframe()

img1 = Image.open(os.path.join("images", df["image_name"][20]))
text1 = df["caption"][20]
img2 = Image.open(os.path.join("images", df["image_name"][10]))
text2 = df["caption"][10]

img1_vec = ret.encode_image(img1)
text1_vec = ret.encode_text(text1)
img2_vec = ret.encode_image(img2)
text2_vec = ret.encode_text(text2)


print("1")
print(torch.cdist(img1_vec, text1_vec))

print("2")
print(torch.cdist(img2_vec, text2_vec))

print(torch.cdist(img1_vec, text2_vec))
print(torch.cdist(img2_vec, text1_vec))
