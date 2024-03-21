import streamlit as st
import torch
import torchvision
import clip
from PIL import Image
import os
from retrival import ClipImgRetrieval, CustImgRetrieval
from data import create_dataframe, create_dataloader, Dataset, image_transform
from model import Model
from config import model_name


print("Loading custom model")
cust_model = Model()
cust_model.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))

print("Loading Clip model")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

print("Loading vector database")
# Pre-load image vectors for efficiency
image_paths = os.listdir("images")[:10000]  # Assuming you have max 10000 images
clip_ret = ClipImgRetrieval(image_paths, clip_model, preprocess, data_file="image_vectors.pt")
cust_ret = CustImgRetrieval(image_paths, cust_model, image_transform, data_file="custom_image_vectors.pt")


# Streamlit App
st.title("CLIP Model Image Retrieval Dashboard")
query = st.text_input("Enter a query:")

clip_col, cust_col = st.columns(2)
clip_col.write("CLIP model")
cust_col.write("Custom model")

if query:
    # Retrieve images based on the query
    results, similarities = clip_ret.retrieve(query)
    for i, image in enumerate(results):
        if i < 5:  # Show only the top 5 images
            clip_col.image(image, width=200)
            clip_col.write(f"{similarities[i]:.4f}")

    results, similarities = cust_ret.retrieve(query)
    for i, image in enumerate(results):
        if i < 5:  # Show only the top 5 images
            cust_col.image(image, width=200)
            cust_col.write(f"{similarities[i]:.4f}")
