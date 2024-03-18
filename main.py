from data import Dataset, create_dataloader, create_dataframe, image_transform
from model import Model
from train import train
from config import model_name

model = Model()

df = create_dataframe(rows=20)
train_ds = Dataset(df["image_name"].to_list(), df["caption"].to_list(), model.text_encoder.tokenizer, image_transform)
train_dl = create_dataloader(train_ds)

train(model, train_dl, epochs=3, lr=0.001, model_name=model_name)
