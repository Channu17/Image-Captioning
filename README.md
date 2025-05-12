# Image Captioning

This project generates captions for images using a deep learning model. It includes a Jupyter notebook for model training, a FastAPI backend to serve the model, and a Streamlit web application for an interactive user interface.

## Features

*   **Image Feature Extraction:** Uses a pre-trained VGG16 model to extract features from images.
*   **Caption Generation:** Employs an LSTM-based model to generate descriptive captions for images.
*   **Training Pipeline:** A Jupyter notebook (`notebooks/main.ipynb`) details the process of downloading the Flickr8k dataset, preprocessing data, training the captioning model, and evaluating it.
*   **API Endpoint:** A FastAPI application (`api.py`) provides an endpoint (`/model`) to get captions for uploaded images.
*   **Interactive Web App:** A Streamlit application (`app.py`) allows users to upload an image and view the generated caption.


The image captioning model consists of an encoder-decoder architecture:

*   **Encoder (Feature Extractor):**
    *   A pre-trained VGG16 model, with its final classification layer removed, is used as the image feature extractor.
    *   It processes an input image (resized to 224x224 pixels) and outputs a 4096-dimensional feature vector representing the image's salient content.
    *   The Keras model for the feature extractor is typically saved as `models/feature_extractor.keras`.

*   **Decoder (Captioning Model):**
    *   An LSTM (Long Short-Term Memory) network serves as the decoder.
    *   It takes the 4096-dimensional image feature vector (from the encoder) and the sequence of previously generated words (as token IDs) as input.
    *   The LSTM then predicts the next word in the caption sequence.
    *   This process is repeated, with the newly predicted word becoming part of the input for the next time step, until an 'endseq' token is generated or the maximum caption length is reached.
    *   The Keras model for the caption generator is typically saved as `models/best_model.keras`.

*   **Text Preprocessing:**
    *   Captions are cleaned (converted to lowercase, punctuation and short words removed).
    *   Special tokens, 'startseq' and 'endseq', are added to mark the beginning and end of each caption, respectively.
    *   A Keras Tokenizer (`models/tokenizer (1).pkl`) is used to convert words into integer sequences and vice-versa. The vocabulary is built from the training captions.

*   **Embeddings:**
    *   Word embeddings are utilized to represent each word in the vocabulary as a dense vector. These embeddings are learned during the training process and are part of the decoder model. They transform the tokenized words into a format suitable for the LSTM layer.

*   **Training:**
    *   The model is trained to minimize the 'categorical_crossentropy' loss function, using the 'adam' optimizer.
    *   During training, the decoder learns to predict the next word in a caption given the image features and the preceding words of the ground-truth caption.


## Technologies Used

*   Python
*   TensorFlow / Keras
*   FastAPI
*   Streamlit
*   Numpy
*   Pillow (PIL)
*   Scikit-learn
*   NLTK (for BLEU score calculation)
*   Kaggle API (for dataset download)
*   Jupyter Notebook

## Project Structure

```
├── Datasets/
│   ├── captions.txt
│   └── Images/
├── models/
│   ├── best_model.keras  # Main captioning model (Decoder)
│   ├── best_model.h5     # Alternative format for the main model
│   ├── feature_extractor.keras # VGG16 based feature extractor (Encoder)
│   ├── feature_extractor.h5    # Alternative format for the feature extractor
│   ├── tokenizer (1).pkl # Tokenizer for caption preprocessing
│   ├── features (1).pkl  # Pre-computed image features (optional, from notebook)
│   ├── model.png         # Diagram of the model architecture (if available)
│   └── ... (other model related files)
├── notebooks/
│   └── main.ipynb        # Jupyter notebook for training and experimentation
├── api.py                # FastAPI backend for serving the model
├── app.py                # Streamlit web application for UI
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <repository-url>
    # cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv imgcap
    source imgcap/bin/activate  # On Windows use `imgcap\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset:**
    *   The Jupyter notebook (`notebooks/main.ipynb`) contains commands to download the Flickr8k dataset using the Kaggle API. You will need to have your `kaggle.json` API key set up.
    *   Alternatively, manually download the Flickr8k dataset and place the `Images` folder and `captions.txt` file into a `Datasets` directory at the root of the project.

5.  **Train the Model (or use pre-trained models):**
    *   Run through the `notebooks/main.ipynb` to train the model. This will generate `best_model.keras`, `feature_extractor.keras`, and `tokenizer.pkl` in the `models` directory (or ensure your pre-trained models are in the `models` directory as specified in `api.py` and `app.py`). The notebook saves models to `/kaggle/working/` by default, so you might need to adjust paths or move the saved models to the `models/` directory in your project. The current `api.py` and `app.py` look for models in a local `models/` directory.

## Usage

### 1. Jupyter Notebook (Training and Exploration)

*   Navigate to the `notebooks/` directory.
*   Open and run `main.ipynb` using Jupyter Notebook or JupyterLab.
    ```bash
    jupyter notebook notebooks/main.ipynb
    # or
    jupyter lab notebooks/main.ipynb
    ```
    This notebook covers:
    *   Downloading and unzipping the Flickr8k dataset.
    *   Extracting image features using VGG16.
    *   Preprocessing captions.
    *   Tokenizing text data.
    *   Building and training the image captioning model (LSTM-based).
    *   Evaluating the model using BLEU scores.
    *   Generating captions for sample images.

### 2. FastAPI Backend

*   To run the API that serves the image captioning model:
    ```bash
    uvicorn api:api --reload
    ```
*   The API will be available at `http://127.0.0.1:8000`.
*   You can access the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.
*   The main endpoint for captioning is `POST /model`, which expects an image file.

### 3. Streamlit Web Application

*   To run the interactive web application:
    ```bash
    streamlit run app.py
    ```
*   The application will typically open in your web browser at `http://localhost:8501`.
*   You can upload an image, and the app will display the image and the generated caption.

## Model Details



This README provides a comprehensive overview of the Image Captioning project.
**APP** : https://image-captioning-17.streamlit.app/
