import torch
import torchvision
import clip
from PIL import Image
import os

class ClipImgRetrieval:
    def __init__(self, image_paths, model, process_image, data_file: str = ""):
        self.image_paths = image_paths
        self.preprocess = process_image
        self.model = model
        if os.path.exists(data_file):
            self.img_vecs = torch.load(data_file)
        else:
            self.img_vecs = []
            for i in range(0, len(self.image_paths)):
                with torch.no_grad():
                    data = self.preprocess(Image.open(os.path.join("images", image_paths[i])).convert("RGB"))
                    temp = data.reshape(1, 3, 224, 224)
                    img_vecs = model.encode_image(temp).float()
                    temp = temp.to('cpu')
                    self.img_vecs.append(img_vecs)
                if i % 200 == 0:
                    print('{}th encoding complete'.format(i))
            self.img_vecs = torch.vstack(self.img_vecs)
            self.img_vecs = self.img_vecs / self.img_vecs.norm(dim=-1, keepdim=True)
            torch.save(self.img_vecs, "image_vectors.pt")

    def encode_text(self, text):
        text_vec = self.model.encode_text(clip.tokenize(text)).float()
        text_vec = text_vec / text_vec.norm(dim=-1, keepdim=True)
        return text_vec

    def encode_image(self, image):
        with torch.no_grad():
            data = self.preprocess(image.convert("RGB"))
            temp = data.reshape(1, 3, 224, 224)
            img_vec = self.model.encode_image(temp).float()
            temp = temp.to('cpu')
        img_vec = img_vec / img_vec.norm(dim=-1, keepdim=True)
        return img_vec

    def retrieve(self, text):
        text_vec = self.model.encode_text(clip.tokenize(text)).float()
        text_vec = text_vec / text_vec.norm(dim=-1, keepdim=True)
        similarities = text_vec @ self.img_vecs.T
        top_5_indices = torch.topk(similarities, k=5, dim=1)[1].squeeze(0)  # Get top 5 indexes
        top_5_values = torch.topk(similarities, k=5, dim=1)[0].squeeze(0)  # Get top 5 indexes
        return [Image.open(os.path.join("images", self.image_paths[i])).convert("RGB") for i in top_5_indices.tolist()], top_5_values.tolist()


class CustImgRetrieval:
    def __init__(self, image_paths, model, process_image, data_file: str = ""):
        self.image_paths = image_paths
        self.preprocess = process_image
        self.model = model
        if os.path.exists(data_file):
            self.img_vecs = torch.load(data_file)
        else:
            self.img_vecs = []
            for i in range(0, len(self.image_paths)):
                with torch.no_grad():
                    data = self.preprocess(Image.open(os.path.join("images", image_paths[i])).convert("RGB"))
                    temp = data.reshape(1, 3, 224, 224)
                    img_vecs = model.encode_image(temp).float()
                    temp = temp.to('cpu')
                    self.img_vecs.append(img_vecs)
                if i % 200 == 0:
                    print('{}th encoding complete'.format(i))
            self.img_vecs = torch.vstack(self.img_vecs)
            self.img_vecs = self.img_vecs / self.img_vecs.norm(dim=-1, keepdim=True)
            torch.save(self.img_vecs, "custom_image_vectors.pt")

    def encode_text(self, text):
        text_vec = self.model.encode_text(text).float()
        text_vec = text_vec / text_vec.norm(dim=-1, keepdim=True)
        return text_vec

    def encode_image(self, image):
        with torch.no_grad():
            data = self.preprocess(image.convert("RGB"))
            temp = data.reshape(1, 3, 224, 224)
            img_vec = self.model.encode_image(temp).float()
            temp = temp.to('cpu')
        img_vec = img_vec / img_vec.norm(dim=-1, keepdim=True)
        return img_vec

    def retrieve(self, text):
        text_vec = self.model.encode_text(text).float()
        text_vec = text_vec / text_vec.norm(dim=-1, keepdim=True)
        similarities = text_vec @ self.img_vecs.T
        top_5_indices = torch.topk(similarities, k=5, dim=1)[1].squeeze(0)  # Get top 5 indexes
        top_5_values = torch.topk(similarities, k=5, dim=1)[0].squeeze(0)  # Get top 5 indexes
        return [Image.open(os.path.join("images", self.image_paths[i])).convert("RGB") for i in top_5_indices.tolist()], top_5_values.tolist()
