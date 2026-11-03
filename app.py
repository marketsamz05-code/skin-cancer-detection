

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Skin Cancer Detection")
st.write("Upload a skin image for prediction.")

# Hugging Face model details
MODEL_ID = "amz110/skin-cancer-model"
MODEL_FILE = "best_skin_cancer_model.keras"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=MODEL_FILE
    )
    return tf.keras.models.load_model(model_path)

CLASS_NAMES = [
    'actinic keratosis',
    'basal cell carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented benign keratosis',
    'seborrheic keratosis',
    'squamous cell carcinoma',
    'vascular lesion'
]

uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Skin Image", width=350)

    if st.button("Predict"):
        try:
            with st.spinner("Analyzing image..."):
                model = load_model()

                resized_image = image.resize((128, 128))
                image_array = np.array(
                    resized_image, dtype=np.float32
                )
                image_array = np.expand_dims(
                    image_array, axis=0
                )

                prediction = model.predict(
                    image_array, verbose=0
                )[0]

                index = int(np.argmax(prediction))

            st.success(
                "Predicted Class: " + CLASS_NAMES[index]
            )

            st.write(
                "Confidence: {:.2f}%".format(
                    prediction[index] * 100
                )
            )

            st.warning(
                "Educational prediction only. "
                "Not a medical diagnosis."
            )

        except Exception as e:
            st.error("Prediction error: " + str(e))
