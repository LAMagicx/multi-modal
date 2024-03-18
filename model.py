import torch.nn.functional as F
from torch import nn, tensor
from transformers import AutoTokenizer, AutoModel
import torchvision
import timm
from config import *
from data import image_transform
from PIL import Image


class ImageEncoder(nn.Module):
    def __init__(self, model_name=image_embedding_model, pretrained=True, trainable=True):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        # self.model = timm.create_model(model_name, pretrained, num_classes=0, global_pool="avg")
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, model_name=text_embedding_model, pretrained=True, trainable=True):
        super().__init__()
        if pretrained:
            self.tokenizer = AutoTokenizer.from_pretrained(text_embedding_model)
            self.model = AutoModel.from_pretrained(text_embedding_model)
        else:
            self.tokenizer = AutoTokenizer.from_config(text_embedding_model)
            self.model = AutoModel.from_config(text_embedding_model)

        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        # tokens = self.tokenizer(text, return_tensors="pt")
        # encoded_text = self.model(**tokens).last_hidden_state
        encoded_text = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return encoded_text[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(image_encoder_size, image_embedding_size)
        self.text_projection = ProjectionHead(text_encoder_size, text_embedding_size)
        self.temperature = model_temperature

    def encode_text(self, text: str = ""):
        tokens = self.text_encoder.tokenizer(text, padding=True, truncation=True, max_length=200, return_tensors='pt')
        text_features = self.text_encoder(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        text_embeddings = self.text_projection(text_features)
        return text_embeddings

    def encode_image(self, image_arr):
        image_features = self.image_encoder(image_arr)
        image_embeddings = self.image_projection(image_features)
        return image_embeddings

    def get_embeddings(self, image: Image = None, image_arr: tensor = None, caption: str = None, input_ids: tensor = None, attention_mask: tensor = None):
        # torch.Size([1, 3, 224, 224])
        # torch.Size([1, 1000])
        # torch.Size([1, 22])
        # torch.Size([1, 768])
        if image is not None and image_arr is None:
            image_arr = image_transform(image.convert('RGB')).unsqueeze(0)
        image_features = self.image_encoder(image_arr)
        if input_ids is not None and attention_mask is not None and caption is None:
            text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            tokens = self.text_encoder.tokenizer(caption, padding=True, truncation=True, max_length=200, return_tensors='pt')
            text_features = self.text_encoder(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        return image_embeddings, text_embeddings

    def forward(self, batch):
        # torch.Size([1, 3, 224, 224])
        # torch.Size([1, 1000])
        # torch.Size([1, 77])
        # torch.Size([1, 768])
        image_features = self.image_encoder(batch['image'])
        text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0
        return loss.mean()
