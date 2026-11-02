
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.title("Skin Cancer Detection")
st.write("Upload a skin image for prediction.")

MODEL_PATH = Path(__file__).parent / "skin_cancer_cnn.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

CLASS_NAMES = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']

uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Skin Image")

    if st.button("Predict"):
        resized_image = image.resize((128, 128))
        image_array = np.array(resized_image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array, verbose=0)[0]
        index = int(np.argmax(prediction))

        st.success("Predicted Class: " + CLASS_NAMES[index])
        st.write(
            "Confidence: {:.2f}%".format(
                prediction[index] * 100
            )
        )

        st.warning(
            "Educational prediction only. Not a medical diagnosis."
        )
