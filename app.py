import streamlit as st
import io
from PIL import Image
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input


model = load_model("models/best_model.keras")
with open("models/tokenizer (1).pkl", "rb") as f:
    tokenizer = pickle.load(f)
feature_extractor = load_model("models/feature_extractor.keras")

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, features, tokenizer, max_length=35):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding="post")
        
        yhat = model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return " ".join(in_text.split()[1:-1])

def preprocess_image(image):
    image = image.resize((224, 224))  
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = feature_extractor.predict(image, verbose=0)
    features = features.reshape(1, -1)
    return features


st.title("🖼️ Image Captioning with AI")
st.write("Upload an image, and the AI will generate a caption for it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            features = preprocess_image(image)
            caption = predict_caption(model, features, tokenizer)
            st.success("Caption: " + caption)
