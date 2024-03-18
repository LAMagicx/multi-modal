"""
Parameters for the model

"""
image_embedding_model = "resnet50"
text_embedding_model = "distilbert-base-uncased"
image_size = 224

image_encoder_size = 1000
text_encoder_size = 768
image_embedding_size = 512
text_embedding_size = 512
model_temperature = 1

batch_size = 64
workers = 1

model_name = "model_512_20k_v2.pt"
