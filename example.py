import torch
import torchvision
import clip
from PIL import Image
import os
import skimage
import IPython.display
import matplotlib.pyplot as plt
import numpy as np

class ImgRetrieval:
    def __init__(self, image_paths, model, process_image):
        self.images = [Image.open(os.path.join("images", image_path)).convert("RGB") for image_path in image_paths]
        self.data = [preprocess(img) for img in self.images]
        self.model = model
        self.img_vecs = []
        for i in range(0, len(self.images)):
            with torch.no_grad():
                temp = self.data[i].reshape(1, 3, 224, 224)
                img_vecs = model.encode_image(temp).float()
                temp = temp.to('cpu')
                self.img_vecs.append(img_vecs)
            if i % 200 == 0:
                print('{}th encoding complete'.format(i))
        self.img_vecs = torch.vstack(self.img_vecs)
        self.img_vecs = self.img_vecs / self.img_vecs.norm(dim=-1, keepdim=True)

    def retrieve(self, text):
        text_vec = self.model.encode_text(clip.tokenize(text)).float()
        text_vec = text_vec / text_vec.norm(dim=-1, keepdim=True)
        pick = torch.argmax(text_vec @ self.img_vecs.T)
        return self.images[pick]


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

tokens = clip.tokenize('Photo of many people near the beach. They are playing with beach ball')

ret = ImgRetrieval(os.listdir("images")[:5000], model, preprocess)
