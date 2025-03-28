from fastapi import FastAPI, HTTPException, File, UploadFile
from tensorflow.keras.models import load_model
import io
from PIL import Image
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

api = FastAPI()
@api.get("/")
def home():
    return {"message" : "Welcome to the image captioning API"}

model = load_model("models/best_model.keras")

with open("models/tokenizer (1).pkl", "rb") as f:
    tokenizer = pickle.load(f)

feature_extractor = load_model("models/feature_extractor.keras")



def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, features, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen = max_length, padding = "post")
        
        yhat = model.predict([features, sequence], verbose = 0)
        yhat = np.argmax(yhat)
        
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " +word
        if word == 'endseq':
            break
    return in_text

def preprocess_image(image):
    image = image.resize((224, 224))  
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)

   # For debugging, add this to your preprocess_image function
    features = feature_extractor.predict(image, verbose=0)
    print("Feature shape before reshape:", features.shape)
    features = features.reshape(1, -1)
    print("Feature shape after reshape:", features.shape)
    return features


@api.post("/model")
async def captioning(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        features = preprocess_image(image)
        max_length = 35
        caption = predict_caption(model, features, tokenizer, max_length)
        caption = caption.split()
        caption = " ".join(caption[1:-1])
        return {"caption":caption}
    except Exception as e:
        return HTTPException(status_code = 500, detail = str(e))
    