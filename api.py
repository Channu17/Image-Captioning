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

def predict_caption(model, image, tokenizer, max_length):
    image = np.expand_dims(image, axis = 0)
    
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.text_to_sequences([in_text])[0]
        sequence = pad_sequences([seqeuence], maxlen = max_length, padding = "post")
        
        yhat = model.predict([image, sequence], verbose = 0)
        yhat = np.argmax(yhat)
        
        word = idx_to_word(yhat, tokenizer)
        if word in None:
            break
        in_text += " " +word
        if word == 'endseq':
            break
    return in_text

def preprocess_iamge(image):
    image= img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    features = feature_extractor.predict(image, verbose = 0)
    return image

@api.get("/model")
async def captioning(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.bytesIO(contents))
        image = preprocess_image(image)
        max_length = 34
        caption = predict_caption(model, image, tokenizer, max_length)
        return {"caption":caption}
    except Exception as e:
        return HTTPExecption(status_code = 500, detail = str(e))
    