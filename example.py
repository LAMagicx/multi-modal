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
    def __init__(self, image_paths, model, process_image, data_file: str = ""):
        self.image_paths = image_paths
        self.preprocess = process_image
        # self.data = [preprocess(Image.open(os.path.join("images", image_path)).convert("RGB")) for image_path in image_paths]
        self.model = model
        if data_file:
            self.img_vecs = torch.load(data_file)
        else:
            self.img_vecs = []
            for i in range(0, len(self.image_paths)):
                with torch.no_grad():
                    data = self.preprocess(Image.open(os.path.join("images", self.image_paths[i])).convert("RGB"))
                    temp = data.reshape(1, 3, 224, 224)
                    img_vecs = model.encode_image(temp).float()
                    temp = temp.to('cpu')
                    self.img_vecs.append(img_vecs)
                if i % 200 == 0:
                    print('{}th encoding complete'.format(i))
            self.img_vecs = torch.vstack(self.img_vecs)
            self.img_vecs = self.img_vecs / self.img_vecs.norm(dim=-1, keepdim=True)
            torch.save(self.img_vecs, "image_vectors.pt")

    def retrieve(self, text):
        text_vec = self.model.encode_text(clip.tokenize(text)).float()
        text_vec = text_vec / text_vec.norm(dim=-1, keepdim=True)
        pick = torch.argmax(text_vec @ self.img_vecs.T)
        return Image.open(os.path.join("images", self.image_paths[pick])).convert("RGB")


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

tokens = clip.tokenize('Photo of many people near the beach. They are playing with beach ball')


ret = ImgRetrieval(image_paths=os.listdir("images")[:10000], model=model, process_image=preprocess, data_file="image_vectors.pt")
