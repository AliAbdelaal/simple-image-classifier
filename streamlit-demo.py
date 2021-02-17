import json
from io import StringIO
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from torch.utils.data import dataset
from classifier import Brain, DataWrapper, train_brain

st.title("Tiny Generic Image Classifier")
brain = Brain()
datasets = DataWrapper.get_datasets()

try:
    brain.load_model()
    st.markdown("A model is available and trained on the following classes\n"+
    "".join([f'- {i}\n' for i in brain.get_supported_classes()]))
except FileNotFoundError:
    st.markdown("> No trained classifier found.")


st.header("Upload your photo here")
uploaded_file = st.file_uploader("choose_file")
col1, col2 = st.beta_columns(2)
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    col1.image(bytes_data)
    if brain.is_trained():
        preds = brain.predict_image(bytes_data)
        col2.json(preds)
    
st.header("train a new classifier !")
data_name = st.selectbox("torchvision data", list(datasets.keys()))
st.write(f'train a new classifier on {data_name} dataset?')
if st.button("confirm"):
    st.info("Model is training")
    brain = train_brain(brain, data_name)
    st.info("Model trained !")
