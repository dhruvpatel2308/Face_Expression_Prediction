import streamlit as st
import requests
import json
from io import BytesIO

st.title('Face Expression Prediction')

image_file = st.file_uploader("Upload Image")

if image_file is not None:
    image = image_file.getvalue()

    response = requests.post(
        "https://dpatel9923-face-expression-prediction.hf.space/prediction",
        files = {
            "image": ByteIO(image)
        }
    )

    label = json.loads(response._content)
    st.write(f"Expression of Face is {label['label']}")
