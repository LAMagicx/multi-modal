import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, AutoModel
import torchvision
import timm
from config import *


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

    def forward(self, batch):
        image_features = self.image_encoder(batch['image'])
        text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        image_embeddings = self.image_projection(image_features).T
        text_embeddings = self.text_projection(text_features).T

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
