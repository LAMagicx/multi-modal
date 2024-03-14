import torch
import os
from PIL import Image
from model import Model
from data import create_dataframe, create_dataloader, Dataset, image_transform
torch.device('cpu')

os.environ["TOKENIZERS_PARALLELISM"] = "true"

df = create_dataframe()

model = Model()
model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))

ds = Dataset("images", df["caption"].to_list(), model.text_encoder.tokenizer, image_transform, rows=100)
# dl = create_dataloader(ds, batch_size=1)


img1 = Image.open(os.path.join("images", df["image_name"][20])).convert("RGB")
img_emb1, text_emb1 = model.get_embeddings(image=img1, caption=df["caption"][20])

img2 = Image.open(os.path.join("images", df["image_name"][10])).convert("RGB")
img_emb2, text_emb2 = model.get_embeddings(image=img2, caption=df["caption"][10])

print("1")
print(torch.cdist(img_emb1, text_emb1))

print("2")
print(torch.cdist(img_emb2, text_emb2))

print(torch.cdist(img_emb1, text_emb2))

print(torch.cdist(img_emb2, text_emb1))
